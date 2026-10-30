# PicturePerfect: Difusion-based Same-identity Facial Replacement

Manik Dhingra<sup>⋆B</sup>, Yashesh Savani<sup>⋆B</sup>, Mahesh Reddy, Antoine Mercier, Jiahe Liu, and Guillaume Berger<sup>B</sup>

Qualcomm AI Research<sup>⋆⋆</sup> {manidhin,ysavani,guilberg}@qti.qualcomm.com

![](images/f75f03c7a3d43a7b75ef56d7893cb69429e50b110140f13fceb4f147124c003d.jpg)  
Fig. 1: We present PicturePerfect, a difusion-based identity preserving facial replacement approach. Prior methods like FaceAdapter [20] tend to alter the facial identity and lose realism on both in-distribution (ID) (left) and out-of-distribution (OOD) (right) face replacement tasks.

Abstract. With the growing success of difusion models in computer vision, we explore their potential for same-identity facial replacement in photographs. Specifically, we propose a difusion-based method, built on top of a pre-trained text-to-image model, that takes as input a portrait image of a person and a second reference image of the same individual, potentially captured under diferent conditions. The goal is to seamlessly replace the input face with the reference face, while keeping the background intact. Surprisingly, despite the clear real-world utility of this task, no recently published work has directly addressed face replacement in this specific setting. To support this goal, we construct a large dataset

of image pairs depicting the same person under varying facial expressions and poses. Experimental results demonstrate that our approach produces more realistic and identity-consistent results than existing face reenactment models.

## 1 Introduction

Pasting a person’s face onto another image and retouching it using traditional image editing software is a challenging task, particularly when the two images difer in viewpoint, illumination, background, hairstyle, or clothing. In this paper, we propose a difusion-based approach, built on top of a pre-trained text-toimage model, that can efectively replace a subject’s face in a photograph with one that matches the pose and expression from another image of the same individual—even in the presence of substantial variation in lighting, pose, or facial expression. A practical application involves replacing an unflattering facial appearance in a group photo with a more favorable one taken from another image, where the person may be smiling or posing more naturally.

An efective solution to this task must preserve the individual’s identity, leave the background unaltered, and seamlessly incorporate the reference face into the original image. This is closely related to face reenactment, where a “source” face is morphed to match the pose and expression of a “driving” face. However, most reenactment methods are designed to support cross-identity generation, where the source and driving images may come from diferent people. While recent reenactment approaches leveraging pre-trained difusion models, such as Face-Adapter [20], produce impressive cross-identity results, we observe that these methods often struggle to produce realistic and identity-preserving outputs that faithfully match the driving image. These methods typically rely on the source image alone to retain identity, which becomes a serious limitation in cases with expressions that deviate significantly from the driving image or extreme poses. As shown in Fig. 1, our method outperforms such approaches by leveraging the reference image more directly — since both images depict the same person, our model can efectively composite the driving face directly into the original photo, rather than attempting to hallucinate it from limited cues.

Our approach shares similarities with early work on automated face swapping [3, 4]. Like our method, these classical approaches rely heavily on the reference image by extracting and “pasting” the reference’s inner face into the original image after performing re-alignment and relighting. However, these methods typically involve complex pipelines composed of hand-engineered components whereas our approach relies on an end-to-end neural network trained specifically for this task.

Our key contributions are as follows:

– We construct a large-scale dataset of image pairs featuring the same individual under varying facial expressions and poses. We intend to release this dataset publicly.

– We propose a simple yet efective training strategy for extending pretrained text-to-image difusion model to same-identity face replacement. Compared to prior works, our approach yields realistic, higher-quality outputs with better identity preservation.

– We demonstrate that methods tailored to same-identity generation significantly outperform existing techniques designed for general-purpose, crossidentity face reenactment.

## 2 Related Work

Same-identity Face Replacement. Same-identity face replacement has recently garnered increased attention, driven by the growing adoption of generative AI in computer vision applications. Notably, several commercial solutions—including Google’s BestTake [18], Oppo’s AI Perfect Shot [40], Huawei’s AI Best Expression [24], and OnePlus’s AI Best Face [39]—incorporate variants of same-identity face replacement. These methods typically utilize a curated set of images of the same individual and perform face replacement. Despite its clear real-world utility, to the best of our knowledge, no recently published work has specifically focused on the challenge of producing high-quality, realistic face replacement where the source and driving face are from the same individual. Instead, existing studies have primarily addressed related tasks, such as face reenactment [5, 22, 52, 65] and face swapping [10, 30, 33, 61, 62, 75].

Face Swapping. Recent face swapping approaches [2, 10, 29, 30, 33, 46, 61, 62, 75] focus on identity swapping, where the identity is altered while preserving pose, expression, and background. Since our work focuses on same-identity face replacement, these methods are related but not directly applicable to our setup.

Face Reenactment. Face reenactment methods are more closely aligned with our task, as they modify a source face to match the pose and expression of a driving image. Most methods are designed for cross-identity generations and therefore rely on intermediate representations that discard the identity of the driving image. Existing techniques can be broadly categorized into warping-based and 3DMM-based approaches. Warping-based methods [22, 23, 50–52, 55, 66, 73] use facial landmarks to transfer motion by warping the source image. These approaches perform well under small motions but degrade significantly with larger pose changes, often producing blurred or distorted outputs due to imprecise motion fields. In contrast, 3DMM-based methods [15, 16, 28, 44, 65] use 3D facial representations [13] and are usually more robust to larger pose variations. However, 3DMMs typically lack high-frequency details, such as hair, teeth, and eye motion. To address this, several works [5, 65] integrate adversarial generators like StyleGAN2 [27]. More recently, difusion-based face reenactment methods have emerged. FADM [67] combines warping-based facial animation approaches [51, 55] with difusion-based refinement. Face-Adapter [20], AniPortrait [57] and DifusionAct [6] also leverage pre-trained difusion models, but rely on keypoints for motion control. A related subcategory includes portrait animation models that require a driving video for motion control. Approaches such as FaceVid2Vid [55], X-Portrait [60], MegActor [63] and LivePortrait [19] fall into this class. Like their image-driven counterparts, these methods are designed for cross-identity reenactment and often struggle to preserve identity when the source image contains extreme expressions or poses. In contrast, our approach focuses on same-identity face replacement and does not require landmarks, 3DMMs, or auxiliary motion representation. Instead, we directly use a distorted version of the driving image itself to control the transformation.

Subject-driven Image Generation. IP-Adapter-FaceID [64] and Instant-ID [54] extend difusion-based text-to-image models for subject-driven generation by conditioning on face identity embeddings extracted from external face encoders. Although efective for tasks such as identity interpolation or novel view synthesis, these methods often fail to reproduce the precise pose or expression of a driving image and may introduce unwanted background changes—limiting their applicability to our use case.

## 3 Method

## 3.1 Learning to Replace Faces in Photographs

Motivation. Our goal is to train a model that can replace a given source face with a driving face of the same individual, potentially captured under diferent conditions, while preserving the original background. Ideally, this task would require {source, driving, target} triplets, where the target image reflects the desired output. However, to the best of our knowledge, no such dataset exists.

Instead, face reenactment models are typically trained on {source, target} pairs extracted from video frames, where the target image also implicitly serves as the driving signal. Using the full target image directly as input would amount to giving the model access to the ground truth, leading to trivial solutions where the model simply reproduces the target image verbatim. To avoid this, existing methods rely on intermediate representations derived from the target, such as keypoints or learned embeddings. These representations intentionally discard most of the low-level information present in the driving image to prevent trivial copying and allow generalization to driving images depicting diferent individuals or captured under diferent conditions.

While such representations are necessary for cross-identity reenactment, they are less suitable for same-identity scenarios. As a result, current face reenactment models must rely heavily on the source image to preserve identity and frequently fail to generate realistic, identity-consistent results that faithfully match the driving image.

Proposed Representation for the Driving Image. To address this limitation, we propose using a distorted version of the inner face region from the target image as the driving signal. This simple yet efective strategy provides strong supervision for preserving the facial attributes of the driving image while preventing the model from trivially copying the target.

![](images/8516b77e7c3d766427b4896e3c75b6315de2bd936c3cd7a622ca3e8dedb83e21.jpg)  
Fig. 2: PicturePerfect network overview: Given a source (s) and target (tg) image pair, we extract the driving image (d) by segmenting the face from the target image. To simulate real-world variations, we apply data augmentations such as scaling, rotation, translation, and relighting. The image triplet is then encoded using the SDXL [42] VAE to obtain latent representations and concatenated spatially.

We begin by segmenting the inner face to exclude the background, ensuring that the model does not rely on the driving signal for this part. This is particularly important for enabling generalization to driving images from external sources, which may contain entirely diferent backgrounds. We then apply a series of augmentations to the segmented face, including color jittering, grayscale conversion, horizontal flipping, and afine transformations (e.g., translation, rotation, and scaling). Additionally, we leverage a relighting module to introduce artificial shadows and lighting variations. These augmentations help the model generalize to images captured under diverse illumination conditions. We feed the obtained distorted face alongside the source image to the model, and train it to reconstruct the original target image from a degraded version of its inner face (and the source image), efectively learning to reverse the applied augmentations. More details on the data augmentation pipeline are provided in Sec. 4.

Model Architecture. Our difusion model, illustrated in Fig. 2, is based on SDXL [42]. To condition the model on both the source and driving images, we concatenate the source image, the distorted driving image, and the noisy latent along the width axis before feeding them into the model, taking inspiration from [35, 49]. We experiment with alternative conditioning strategies, including channel-wise concatenation and ControlNet-style [69] adapter modules, but found spatial concatenation to yield the best performance for this task.

## 3.2 The PicturePerfect Dataset

Existing Datasets Most face reenactment models are trained using videobased face datasets, where two frames are randomly sampled to serve as source and target images. A common choice are the VoxCeleb datasets (VoxCeleb1 [37] and VoxCeleb2 [11]), which consist of interview videos crawled from YouTube. However, VoxCeleb was originally created for speech-related tasks, where visual fidelity is not a priority. As a result, a significant portion of the VoxCeleb videos are low-resolution, and most clips depict individuals talking, with minimal variation in pose or expression throughout. Consequently, randomly sampling two frames often results in pairs with similar facial attributes.

Although more recent datasets such as TalkingHead-1KH [55], VFHQ [59] and CelebV-HQ [74] ofer higher visual quality, they remain heavily biased toward “talking heads” footage crawled from YouTube. Other datasets, such as MEAD [53], FaceForensics [47], and UvA-NEMO [14], are typically reserved for evaluation due to their smaller size or lack of identity diversity.

Proposed Dataset. To address these limitations, we introduce the PicturePerfect dataset—a large-scale collection comprising approximately 500,000 images spanning over 6,724 unique identities, and from which more than 10 million distinct source–target pairs can be sampled. Sample images from our dataset are shown in Fig. 3. Each face in our dataset is available under 21 distinct poses and expressions, including a neutral image (row 1) and 20 expressive variants generated using a face animation model (rows 2-4). To construct the dataset, we leverage an internal face video dataset and LivePortrait [19], a model capable of animating a face based on a driving video. We begin by extracting 25,000 neutral faces from our internal dataset, each depicting a person facing the camera with a neutral expression and direct eye contact. We then manually select 200 neutral-to-expressive driving videos, chosen to cover a wide range of facial expressions and head poses. Each neutral face is animated 20 times, each time using a randomly selected driving video, and we retain only the final frame of each animation. This yields 20 expressive images sharing the same background and visual identity as the neutral source.

All images are standardized to high resolution (1024 × 1024). We further generate descriptive captions for each image using LLaVA v1.5–7B [32]. During training, we randomly sample two images from these sets of 21 (1 neutral + 20 expressive) to form source–target pairs. From each set of 21 images, up to 420 unique source–target combinations can be sampled (excluding self-pairs). With 25,000 such sets, this results in over 10 million unique source–target pairs available for training.

Our dataset captures broad diversity across gender identities, skin tones, and environmental conditions—from dimly lit settings to brightly illuminated scenes, and from close-ups to wide shots. Examples of driving videos used in animation are illustrated in the supplementary material. These include a variety of expressions (e.g., “make a funny face”, “smile with mouth opened”, “stick your tongue out”, “make a sad face”) and head movements (e.g., “look left”, “tilt your head upward”).

![](images/74305fc737391d01efa4cec1f8ddc44be972b30173cbbab063a077280ca545f6.jpg)  
Fig. 3: PicturePerfect dataset samples. For each neutral face (top row), we generate 20 expressive variants using a face animation model [19]. For visualization purposes, we display three randomly sampled expressive variants per neutral face.

Compared to sampling two random frames from the same video, our synthetic data generation pipeline allows finer control over dataset composition. For instance, we deliberately oversample driving videos featuring subtle variations of smiling, resulting in expressive faces that appear to be “posing for the camera”—a scenario that is particularly useful for photographic applications such as replacing an unflattering expression in a group photo with a more favorable one from another image. Such “posed” expressions are rare in existing video-based face datasets, which are typically crawled from YouTube interviews and dominated by conversational or neutral expressions. Another advantage is that, during training, source–target pairs can be selected from dissimilar driving videos to increase task dificulty and model robustness.

## 4 Experiments

## 4.1 Implementation Details

Training Hyperparameters. We fine-tune the pre-trained SDXL model [42] for 25,000 iterations on our proposed dataset (Sec. 3.2). Training is performed

FADM

Source  
Target  
Ours  
DAM  
![](images/3b7236deb1f0d919ac8cc70834e3a4f247d3461ab12d24971de17c63cd3d1f67.jpg)  
Fig. 4: Face Replacement (in-distribution). Our method produces high-quality, realistic reconstructions with stronger identity preservation by leveraging both the source and masked driving image (red inset), outperforming other approaches for diverse identities.

Source  
Target  
Ours  
FaceAdapter  
DAM  
FADM  
HyperReenact  
IP-Adapter  
![](images/fb92fce3954ba3c576a64afdb2c0b9d36f09b1ec16b42baff57fc90f6a74c4ca.jpg)  
Fig. 5: Face Replacement (out-of-distribution). Our method demonstrates strong generalization for a wide range of facial poses, expressions, identities, and lighting conditions. Moreover, it efectively generates reenacted source images that preserve the subject’s identity. Rows 1-3 showcase results on authentic image-pairs from the PicturePerfect test-set, while row 4 contains results on synthetic images.

on 8 NVIDIA A100 GPUs with an efective batch size of 64. We use the Adam optimizer with a learning rate of $1 \times 1 0 ^ { - 5 } , \beta _ { 1 } = 0 . 9$ , and $\beta _ { 2 } = 0 . 9 9 9$

Data Augmentation. Before fine-tuning, we extract segmentation masks for the target image using SegFormer [58]. During training, these masks are used to isolate the inner face region and exclude the background. To improve robustness, we randomly include optional regions such as hair and neck in the mask. The resulting mask is further refined by applying morphological erosion with a kernel size randomly sampled from the range [0, 30], helping to remove background spillover near the face boundary. To make the model robust to varying lighting conditions, we relight the inner face using NeuralGafer [25], with environment lighting sourced from the PolyHaven dataset [43]. The relit inner face is then subjected to a series of augmentations. Finally, the source image, the augmented inner face, and the target image are resized to 576 × 576 and encoded using the VAE module [42] associated with SDXL. These representations are concatenated along the spatial axis and provided as input to the model. During training, captions are randomly dropped for 10% of the samples to improve robustness to missing text conditioning.

Table 1: Quantitative comparison with prior approaches using ArcFace CSIM [12], FID, pose (APD) [31], expression (AED) [31], gaze (AGD) [5] errors, LPIPS [70], and PSNR. Best scores are highlighted in blue, and second-best are underlined. FFHQ and VoxCeleb2 samples are generated with LivePortrait, while PicturePerfect uses curated real-world pairs. Baselines with unreliable FID scores are marked as (—).  
Replacement in-distribution (ID)
<table><tr><td colspan="8">FFHQ + VoxCeleb2 (LivePortrait)</td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td>DAM</td><td>0.500</td><td>24.86</td><td>2.90</td><td>0.2604</td><td>12.45</td><td>0.0362</td><td>21.66</td></tr><tr><td>FADM</td><td>0.477</td><td>48.42</td><td>2.98</td><td>0.2643</td><td>13.41</td><td>0.0455</td><td>20.09</td></tr><tr><td>IP-Adapter</td><td>0.282</td><td>19.28</td><td>3.07</td><td>0.2804</td><td>18.19</td><td>0.0886</td><td>15.40</td></tr><tr><td>FaceAdapter</td><td>0.369</td><td></td><td>2.99</td><td>0.2636</td><td>11.64</td><td>0.1127</td><td>10.43</td></tr><tr><td>HyperReenact</td><td>0.405</td><td></td><td>3.04</td><td>0.2914</td><td>11.22</td><td>0.1038</td><td>13.69</td></tr><tr><td>LivePortrait</td><td>0.558</td><td></td><td>2.48</td><td>0.2740</td><td>8.49</td><td>0.0391</td><td>20.30</td></tr><tr><td>Ours</td><td>0.818</td><td>7.81</td><td>2.81</td><td>0.2592</td><td>5.95</td><td>0.0640</td><td>19.17</td></tr></table>

<table><tr><td colspan="10">Replacement out-of-distribution (OOD)</td></tr><tr><td></td><td></td><td>PicturePerfect (Real)</td><td></td><td></td><td>VoxCeleb2</td><td colspan="4">(LivePortrait)</td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>CSIM↑</td><td>FID↓ AED↓</td><td>APD↓</td><td>AGD↓</td></tr><tr><td>DAM</td><td>0.350</td><td>55.16</td><td>2.61</td><td>0.2222</td><td>15.29</td><td>0.246</td><td>38.51 2.91</td><td>0.2472</td><td>15.59</td></tr><tr><td>FADM</td><td>0.342</td><td>74.98</td><td>2.66</td><td>0.2252</td><td>16.51</td><td>0.219</td><td>61.76 3.01</td><td>0.2490</td><td>16.45</td></tr><tr><td>IP-Adapter</td><td>0.275</td><td>39.16</td><td>2.45</td><td>0.2666</td><td>16.59</td><td>0.214</td><td>23.68 3.14</td><td>0.2761</td><td>23.13</td></tr><tr><td>FaceAdapter</td><td>0.343</td><td></td><td>2.53</td><td>0.2076</td><td>11.31</td><td>0.207</td><td>3.06</td><td>0.3104</td><td>14.61</td></tr><tr><td>HyperReenact</td><td>0.359</td><td></td><td>2.25</td><td>0.1999</td><td>11.52</td><td>0.275</td><td>3.34</td><td>0.3235</td><td>11.99</td></tr><tr><td>Ours</td><td>0.631</td><td>28.08</td><td>2.50</td><td>0.2262</td><td>9.87</td><td>0.579</td><td>12.44 2.78</td><td>0.2366</td><td>12.78</td></tr></table>

Evaluation Datasets. Following standard practices in face reenactment, we evaluate our model under two settings: (1) in-distribution (ID), where the driving and source images are captured moments apart—for example, in burst-mode photo sequences— and (2) out-of-distribution (OOD), where the driving image shows the same person under diferent conditions, for instance when pictures are taken at diferent times under varying background and lighting conditions. For the replacement-ID setting, we use 5,000 test images each from FFHQ and Vox-Celeb2, all of which are synthesized using the LivePortrait animation pipeline to ensure consistency and controlled variations. As for the replacement-OOD setting, we use 5,000 test samples each from PicturePerfect (Sec. 3.2) and Vox-Celeb2. To generate the PicturePerfect test-set on the OOD setup, instead of using LivePortrait, two random frames were captured from two separate video submissions by the same crowd-worker to evaluate model robustness and generalizability on real-world data.

![](images/e35aec98d0e810f7f1847a9b78d38d620573eb2826187fac3ba8f98858316d5b.jpg)

![](images/f607a7d3fc46275c852d174175b29b98f32a58c83ab44eeb85604fae6e317cee.jpg)  
Fig. 6: User study: Replacement ID and OOD. Participants were asked to mark {source, output} pairs as “real” or “fake” based on perceptual quality. We report the proportion of images marked as “real” for each approach. The error bars represent the standard error of the mean (SEM).

Baselines. We compare our model against five face reenactment approaches: DAM [52], FADM [67], Face-Adapter [20], HyperReenact [5], and LivePortrait [19], as well as one subject-driven image generation baseline, based on IP-Adapter [64]. LivePortrait is excluded from the OOD setting due to the absence of driving videos compatible with this setup. In the ID setting, we simulate a two-frame video by treating the source and driving images as sequential frames. For the IP-Adapter baseline, we use a dual-adapter SDXL configuration, with each adapter conditioned on a CLIP embedding of the source image and driving image respectively (more details are available in the supplementary material). For the other baselines, we use the oficial implementations provided by the authors.

Metrics. We evaluate identity preservation using CSIM, overall image quality with PSNR, LPIPS, and FID [21]. Following prior work on face reenactment and portrait animation, we additionally report Average Expression Distance (AED) [31], Average Pose Distance (APD) [31], and Average Gaze Distance (AGD) [5], which measure discrepancies in facial expression, head pose, and gaze direction between generated and target images, respectively.

## 4.2 Results

Replacement in-distribution (ID). Fig. 4 shows a qualitative comparison of replacement-ID results across all methods. Our approach achieves higher reconstruction quality in most examples, producing identity-consistent outputs. In contrast, methods such as DAM, FADM, Face-Adapter, HyperReenact, and LivePortrait primarily rely on the source image for identity, resulting in poor reconstruction quality when the source is in a non-frontal pose (e.g., rows 1 and 3). IP-Adapter, meanwhile, struggles to produce realistic outputs that accurately match the driving image and often fails to preserve the original background.

Tab. 1 reports quantitative comparisons of replacement-ID performance. Our method outperforms prior approaches on most metrics, with particularly strong results in CSIM, FID, and AGD. Although some baselines achieve better scores on LPIPS and PSNR, these metrics often favor overly smooth or blurry reconstructions over perceptual fidelity (see supplementary material for more details).

Replacement out-of-distribution (OOD). The results for visual comparison of the replacement-OOD are provided in Fig. 5. Compared to prior methods, our approach consistently achieves identity-preserving outputs, even under challenging conditions such as exaggerated expressions in the driving images. Generative baselines like FaceAdapter, IP-Adapter, and HyperReenact struggle with identity preservation and often introduce undesirable alterations to the background of the source image. Landmark-based methods such as DAM and FADM are limited by the low identity information contained in facial landmarks and also face dificulties handling large motion diferences between the source and driving images, resulting in blurry or low-quality outputs.

As shown in Tab. 1, our proposed method quantitatively outperforms all baseline approaches across key evaluation metrics, demonstrating its efectiveness in leveraging facial features from the driving image for accurate face replacement. Furthermore, results on the OOD setup showcase that our solution generalizes well on both synthetic as well as real-world data.

![](images/16f22eef67a822494df9cdd7d2e22f9cba83a76ecac4790c52607a0ad114cb93.jpg)  
Source

![](images/1f006a47fbb821d79ba85f9ee975a635950e2e8e9c68caa99fd95de1ded2751d.jpg)  
Driving

![](images/0b033d4a8c8e4ed0e2cbdc7821940fc98fd1c9129b0f60dfc01b1c2f01a6a37f.jpg)  
Ours

![](images/3787528f3101f20e4a20428829368d8acd8b4bd0b165c15890d22732a0089245.jpg)

![](images/974cc5cf284becaff36e9f71cc354a0c9c449c4cf9dce598fb633d7baa1f03e9.jpg)  
No Relighting

![](images/52a05fd698d36bb759db3ebf0592fdf0415cb41b07e6355dfd75fddfdac5480f.jpg)  
Fig. 7: Ablation study: Efects of Relighting. Without relighting, lighting artifacts from the driving image often leak into the output. Incorporating random relighting into the data augmentation pipeline produces significantly more realistic and consistent illumination.

![](images/dae2b38bf5b2597082afb59ab6eba1d7c4f5837a9e8e262b0bfc19515a52c59e.jpg)  
Fig. 8: Ablation study. We evaluate replacing spatial to channel-wise concatenation, swapping SDXL to SDv1.5, as well as disabling relighting, all data augmentations and inner face extraction.

User Study. As part of our quantitative evaluation, we conducted a user study involving ten participants. Each participant reviewed a total of 252 and 175 outputs for ID and OOD face replacement tasks respectively. For each example, participants were shown the source image alongside the model’s output and asked to label the result as either “real” or “fake” based on photorealism and perceptual quality. As shown in Fig. 6, participants judged our model’s outputs to be realistic in 59% of replacement-ID cases and 44% of replacement-OOD cases—substantially outperforming other baselines.

## 4.3 Ablation Study

We conduct a series of ablation studies to evaluate key components of our proposed pipeline. Fig. 8 shows visual comparisons for diferent configurations, and quantitative results for a subset of ablations are reported in Tab. 2. The quantitative metrics for the remaining ablation studies were omitted because the numerical results were counter-intuitive when compared to their corresponding qualitative outputs (as shown in Fig. 8). The visual quality of the ablation variants is clearly inferior to that of our final model, however, when computing standard quantitative metrics on these outputs, the scores misleadingly suggested better performance than the proposed solution—an inconsistency that does not align with the observed qualitative degradation. We further discuss this sensitivity and the limitations of current quantitative metrics relative to perceptual quality in the supplementary material.

Our key findings are as follows: Using the target image directly as the driving input results in a degenerate solution in which the model simply reproduces the driving image. Removing all data augmentations severely degrades performance, efectively pasting the driving face onto the source with minimal adjustment in color or structure. Removing relighting introduces “lighting leakage” artifacts—i.e., the model fails to adjust illumination and retains lighting efects from the driving image. Additional results for the no-relighting condition are shown in Fig. 7. Channel-wise concatenation leads to consistently poorer performance across all metrics and produces visual artifacts, particularly around fine details such as the teeth. As expected, using SDv1.5 [45] results in slightly lower overall performance. While the SDv1.5-based variant occasionally preserves the pose of the source image more accurately, it tends to produce oversaturated color compositions and lower-quality facial details.

Table 2: Ablation study. We compare spatial versus channel-wise concatenation of the input images and assess the performance with SDv1.5 as the backbone UNet architecture [45]. We report the main ablation results here and defer the remaining variants to the supplementary material.
<table><tr><td colspan="8">Replacement in-distribution (ID)</td></tr><tr><td></td><td colspan="3">FFHQ + VoxCeleb2 (LivePortrait)</td><td colspan="3"></td><td></td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td>Ours</td><td>0.818</td><td>7.81</td><td>2.81</td><td>0.2592</td><td>5.95</td><td>0.0640</td><td>19.17</td></tr><tr><td>Channel-wise</td><td>0.689</td><td>55.19</td><td>3.67</td><td>0.3542</td><td>6.18</td><td>0.0831</td><td>16.11</td></tr><tr><td>SDv1.5</td><td>0.776</td><td>9.90</td><td>2.87</td><td>0.2620</td><td>6.58</td><td>0.0564</td><td>17.55</td></tr></table>

<table><tr><td colspan="11">Replacement out-of-distribution (OOD)</td></tr><tr><td colspan="7">PicturePerfect (Real)</td><td colspan="4">VoxCeleb2 (LivePortrait)</td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td></tr><tr><td>Ours</td><td>0.631</td><td>28.08</td><td>2.50</td><td>0.2262</td><td>9.87</td><td>0.579</td><td>12.44</td><td>2.78</td><td>0.2366</td><td>12.78</td></tr><tr><td>Channel-wise</td><td>0.618</td><td>80.70</td><td>3.37</td><td>0.3113</td><td>8.47</td><td>0.548</td><td>62.26</td><td>3.69</td><td>0.3418</td><td>11.80</td></tr><tr><td>SDv1.5</td><td>0.655</td><td>30.98</td><td>2.50</td><td>0.2164</td><td>9.71</td><td>0.576</td><td>12.96</td><td>2.84</td><td>0.2413</td><td>13.19</td></tr></table>

## 5 Discussion and Conclusion

We have introduced PicturePerfect, a difusion-based method for realistic, sameidentity facial replacement in photographs. Our approach leverages a pre-trained text-to-image model and a simple but efective training strategy: by feeding the model a distorted version of the target image as the driving signal, we encourage it to rely on the driving image’s expression and pose. As a result, the model learns to composite the driving face on the source image while preserving the subject’s identity and background. Experimental results demonstrate that our method outperforms existing face reenactment and subject-driven generation techniques on both in- and out-of-distribution face replacement settings. A key part of this work is the PicturePerfect dataset, a large-scale collection of high quality same-identity image pairs that exhibit diverse facial expressions, poses, and illumination conditions. To support further research, we will release this dataset publicly. A particularly compelling application of our method, illustrated in Fig. 9, lies in real-world composite group photo editing, where users can replace unflattering expressions in a group shot with better alternatives from other photos of the same individual, allowing a level of post hoc control dificult to achieve with prior approaches.

![](images/86deb0da1e256e6bbf27a8c6ffe355a39f47fcf5fd103bc05fbdfb8229fe679c.jpg)  
Fig. 9: Real-world simulation of composite picture editing (“burst” mode reconstruction). Each one of the source faces is individually edited using the corresponding target face as the driving image. The model automatically adjusts for orientation, head pose, and blending issues to naturally fit the source composite photograph.

Although our method produces high-quality and perceptually realistic results, it is not without limitations. Despite demonstrating strong generalization on the OOD face replacement task, the predominance of LivePortrait-generated data in the training set may bias the model toward the LivePortrait data distribution. Artifacts may appear in fine details such as hair or around the eyes, and performance degrades when the driving face is extremely small or large, likely due to encoding challenges. An interesting future research direction could explore better head pose control by conditioning the model on keypoints or learned facial embeddings. Because our model is trained specifically for same-identity face replacement, it is not designed to handle cross-identity scenarios. Nonetheless, we include qualitative results in the supplementary material to illustrate how the model behaves in such cases: some outputs transfer identity, while others transfer only expression. These observations highlight interesting directions for future work on disentangling identity and expression.

## References

1. Avrahami, O., Fried, O., Lischinski, D.: Blended latent difusion. TOG (2023)

2. Baliah, S., Lin, Q., Liao, S., Liang, X., Khan, M.H.: Realistic and eficient face swapping: A unified approach with difusion models. In: 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV). pp. 1062–1071. IEEE (2025)

3. Bitouk, D., Kumar, N., Dhillon, S., Belhumeur, P., Nayar, S.K.: Face swapping: automatically replacing faces in photographs. In: ACM SIGGRAPH 2008 papers, pp. 1–8 (2008)

4. Blanz, V., Scherbaum, K., Vetter, T., Seidel, H.P.: Exchanging faces in images. Computer Graphics Forum 23 (2004), https://api.semanticscholar.org/ CorpusID:2803614

5. Bounareli, S., Tzelepis, C., Argyriou, V., Patras, I., Tzimiropoulos, G.: Hyperreenact: one-shot reenactment via jointly learning to refine and retarget faces. In: ICCV (2023)

6. Bounareli, S., Tzelepis, C., Argyriou, V., Patras, I., Tzimiropoulos, G.: Difusionact: Controllable difusion autoencoder for one-shot face reenactment. arXiv preprint arXiv:2403.17217 (2024)

7. Brack, M., Friedrich, F., Kornmeier, K., Tsaban, L., Schramowski, P., Kersting, K., Passos, A.: Ledits++: Limitless image editing using text-to-image models. In: CVPR (2024)

8. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: CVPR (2023)

9. Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., Zheng, Y.: Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In: ICCV (2023)

10. Chen, R., Chen, X., Ni, B., Ge, Y.: Simswap: An eficient framework for high fidelity face swapping. In: Proceedings of the 28th ACM international conference on multimedia. pp. 2003–2011 (2020)

11. Chung, J.S., Nagrani, A., Zisserman, A.: Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622 (2018)

12. Deng, J., Guo, J., Yang, J., Xue, N., Kotsia, I., Zafeiriou, S.: Arcface: Additive angular margin loss for deep face recognition (2022)

13. Deng, Y., Yang, J., Xu, S., Chen, D., Jia, Y., Tong, X.: Accurate 3d face reconstruction with weakly-supervised learning: From single image to image set. In: CVPRW (2019)

14. Dibeklioğlu, H., Salah, A.A., Gevers, T.: Are you really smiling at me? spontaneous versus posed enjoyment smiles. In: ECCV (2012)

15. Ding, Z., Zhang, X., Xia, Z., Jebe, L., Tu, Z., Zhang, X.: Difusionrig: Learning personalized priors for facial appearance editing. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12736–12746 (2023)

16. Doukas, M.C., Zafeiriou, S., Sharmanska, V.: Headgan: One-shot neural head synthesis and editing. In: ICCV (2021)

17. Esler, T.: Face recognition using pytorch. https : / / github . com / timesler / facenet-pytorch, accessed: 2025-04-25

18. Google: Google best take (2025), https://store.google.com/intl/en/ideas/ articles/pixel-best-take/

19. Guo, J., Zhang, D., Liu, X., Zhong, Z., Zhang, Y., Wan, P., Zhang, D.: Liveportrait: Eficient portrait animation with stitching and retargeting control. arXiv preprint arXiv:2407.03168 (2024)

20. Han, Y., Zhu, J., He, K., Chen, X., Ge, Y., Li, W., Li, X., Zhang, J., Wang, C., Liu, Y.: Face-adapter for pre-trained difusion models with fine-grained id and attribute control. In: ECCV (2024)

21. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium (2017)

22. Hong, F.T., Xu, D.: Implicit identity representation conditioned memory compensation network for talking head video generation. In: ICCV (2023)

23. Hong, F.T., Zhang, L., Shen, L., Xu, D.: Depth-aware generative adversarial network for talking head video generation. In: CVPR (2022)

24. Huawei: Huawei ai best expression (2025), https://consumer.huawei.com/my/ community/details/topicId-184693/

25. Jin, H., Li, Y., Luan, F., Xiangli, Y., Bi, S., Zhang, K., Xu, Z., Sun, J., Snavely, N.: Neural gafer: Relighting any object via difusion (2024)

26. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks. In: CVPR (2019)

27. Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., Aila, T.: Analyzing and improving the image quality of stylegan. In: CVPR (2020)

28. Kim, H., Garrido, P., Tewari, A., Xu, W., Thies, J., Niessner, M., Pérez, P., Richardt, C., Zollhöfer, M., Theobalt, C.: Deep video portraits. ACM TOG (2018)

29. Kim, K., Kim, Y., Cho, S., Seo, J., Nam, J., Lee, K., Kim, S., Lee, K.: Difface: Difusion-based face swapping with facial guidance. Pattern Recognition 163, 111451 (2025)

30. Li, L., Bao, J., Yang, H., Chen, D., Wen, F.: Faceshifter: Towards high fidelity and occlusion aware face swapping. arXiv preprint arXiv:1912.13457 (2019)

31. Lin, C.Z., Lindell, D.B., Chan, E.R., Wetzstein, G.: 3d gan inversion for controllable portrait image animation. arXiv preprint arXiv:2203.13441 (2022)

32. Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: CVPR (2024)

33. Liu, Z., Li, M., Zhang, Y., Wang, C., Zhang, Q., Wang, J., Nie, Y.: Fine-grained face swapping via regional gan inversion. In: CVPR (2023)

34. Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.Y., Ermon, S.: Sdedit: Guided image synthesis and editing with stochastic diferential equations. In: ICLR (2022)

35. Mercier, A., Nakhli, R., Reddy, M., Yasarla, R., Cai, H., Porikli, F., Berger, G.: Hexagen3d: stabledifusion is just one step away from fast and diverse text-to-3d generation. arXiv preprint arXiv:2401.07727 (2024)

36. Mokady, R., Hertz, A., Aberman, K., Pritch, Y., Cohen-Or, D.: Null-text inversion for editing real images using guided difusion models. In: CVPR (2023)

37. Nagrani, A., Chung, J.S., Zisserman, A.: Voxceleb: a large-scale speaker identification dataset. arXiv preprint arXiv:1706.08612 (2017)

38. Nichol, A., Dhariwal, P., Ramesh, A., Shyam, P., Mishkin, P., McGrew, B., Sutskever, I., Chen, M.: Glide: Towards photorealistic image generation and editing with text-guided difusion models. arXiv preprint arXiv:2112.10741 (2021)

39. OnePlus: Oneplus ai best face (2025), https://www.oneplus.com/global/nord-4

40. Oppo: Oppo ai perfect shot (2025), https://www.oppo.com/en/smartphones/ series-reno/reno14/

41. Parmar, G., Kumar Singh, K., Zhang, R., Li, Y., Lu, J., Zhu, J.Y.: Zero-shot image-to-image translation (2023)

42. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. In: ICLR (2024)

43. PolyHaven: Polyhaven (2025), https://polyhaven.com/

44. Ren, Y., Li, G., Chen, Y., Li, T.H., Liu, S.: Pirenderer: Controllable portrait image generation via semantic neural rendering. In: ICCV (2021)

45. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022)

46. Roop-Unleased: Roop unleashed: Free face swap tool for images & videos (2025), https://www.roopunleashed.com/

47. Rössler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., Nießner, M.: Faceforensics: A large-scale video dataset for forgery detection in human faces. arXiv preprint arXiv:1803.09179 (2018)

48. Seedream, T., Chen, Y., Gao, Y., Gong, L., Guo, M., Guo, Q., Guo, Z., Hou, X., Huang, W., Huang, Y., et al.: Seedream 4.0: Toward next-generation multimodal image generation. arXiv preprint arXiv:2509.20427 (2025)

49. Shi, Y., Wang, P., Ye, J., Long, M., Li, K., Yang, X.: Mvdream: Multi-view difusion for 3d generation. arXiv preprint arXiv:2308.16512 (2023)

50. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: Animating arbitrary objects via deep motion transfer. In: CVPR (2019)

51. Siarohin, A., Lathuilière, S., Tulyakov, S., Ricci, E., Sebe, N.: First order motion model for image animation (2019)

52. Tao, J., Wang, B., Xu, B., Ge, T., Jiang, Y., Li, W., Duan, L.: Structure-aware motion transfer with deformable anchor model. In: CVPR (2022)

53. Wang, K., Wu, Q., Song, L., Yang, Z., Wu, W., Qian, C., He, R., Qiao, Y., Loy, C.C.: Mead: A large-scale audio-visual dataset for emotional talking-face generation. In: ECCV (2020)

54. Wang, Q., Bai, X., Wang, H., Qin, Z., Chen, A.: Instantid: Zero-shot identitypreserving generation in seconds. arXiv preprint arXiv:2401.07519 (2024)

55. Wang, T.C., Mallya, A., Liu, M.Y.: One-shot free-view neural talking-head synthesis for video conferencing. In: CVPR (2021)

56. Wei, C., Xiong, Z., Ren, W., Du, X., Zhang, G., Chen, W.: Omniedit: Building image editing generalist models through specialist supervision. In: ICLR (2024)

57. Wei, H., Yang, Z., Wang, Z.: Aniportrait: Audio-driven synthesis of photorealistic portrait animation. arXiv preprint arXiv:2403.17694 (2024)

58. Xie, E., Wang, W., Yu, Z., Anandkumar, A., Alvarez, J.M., Luo, P.: Segformer: Simple and eficient design for semantic segmentation with transformers (2021)

59. Xie, L., Wang, X., Zhang, H., Dong, C., Shan, Y.: Vfhq: A high-quality dataset and benchmark for video face super-resolution. In: CVPRW (2022)

60. Xie, Y., Xu, H., Song, G., Wang, C., Shi, Y., Luo, L.: X-portrait: Expressive portrait animation with hierarchical motion attention. In: ACM TOG (2024)

61. Xu, C., Zhang, J., Hua, M., He, Q., Yi, Z., Liu, Y.: Region-aware face swapping. In: CVPR (2022)

62. Xu, Z., Hong, Z., Ding, C., Zhu, Z., Han, J., Liu, J., Ding, E.: Mobilefaceswap: A lightweight framework for video face swapping. In: AAAI (2022)

63. Yang, S., Li, H., Wu, J., Jing, M., Li, L., Ji, R., Liang, J., Fan, H.: Megactor: Harness the power of raw video for vivid portrait animation. arXiv preprint arXiv:2405.20851 (2024)

64. Ye, H., Zhang, J., Liu, S., Han, X., Yang, W.: Ip-adapter: Text compatible image prompt adapter for text-to-image difusion models (2023)

65. Yin, F., Zhang, Y., Cun, X., Cao, M., Fan, Y., Wang, X., Bai, Q., Wu, B., Wang, J., Yang, Y.: Styleheat: One-shot high-resolution editable talking face generation via pre-trained stylegan. In: ECCV (2022)

66. Zakharov, E., Shysheya, A., Burkov, E., Lempitsky, V.: Few-shot adversarial learning of realistic neural talking head models. In: ICCV (2019)

67. Zeng, B., Liu, X., Gao, S., Liu, B., Li, H., Liu, J., Zhang, B.: Face animation with an attribute-guided difusion model. In: CVPR (2023)

68. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing (2023)

69. Zhang, L., Rao, A., Agrawala, M.: Adding conditional control to text-to-image difusion models. In: ICCV (2023)

70. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: CVPR (2018)

71. Zhang, S., Yang, X., Feng, Y., Qin, C., Chen, C.C., Yu, N., Chen, Z., Wang, H., Savarese, S., Ermon, S., et al.: Hive: Harnessing human feedback for instructional visual editing. In: CVPR (2024)

72. Zhao, H., Ma, X.S., Chen, L., Si, S., Wu, R., An, K., Yu, P., Zhang, M., Li, Q., Chang, B.: Ultraedit: Instruction-based fine-grained image editing at scale (2024)

73. Zhao, J., Zhang, H.: Thin-plate spline motion model for image animation. In: CVPR (2022)

74. Zhu, H., Wu, W., Zhu, W., Jiang, L., Tang, S., Zhang, L., Liu, Z., Loy, C.C.: CelebV-HQ: A large-scale video facial attributes dataset. In: ECCV (2022)

75. Zhu, Y., Li, Q., Wang, J., Xu, C.Z., Sun, Z.: One shot face swapping on megapixels. In: CVPR (2021)

## A Supplementary Material

## A.1 Additional Implementation Details

Training. As part of our data augmentation pipeline, we extract the inner face from driving images and apply relighting. To achieve this, we use Seg-Former [58] to generate face masks and NeuralGafer [25] to perform relighting, with background lighting sourced from randomly selected images in the Poly-Haven dataset [43]. To reduce training time and GPU memory usage, both operations are performed ofline prior to training. More specifically, for each source image, we generate and save six driving images—five relighted variants and one original—as well as a single face mask. During training, one of the six driving images is randomly sampled for use in each iteration.

Inference. Our results are generated using 50 denoising steps and a guidance scale of 1.5, and takes ∼ 3 seconds per 576 × 576 image on a single NVIDIA A100. For classifier-free guidance, we drop the text prompt and use a zeroed-out version of the driving image for the negative branch.

Baselines. For most baselines, we use the oficial implementations provided by the respective authors: DAM<sup>1</sup>, FADM<sup>2</sup>, HyperReenact<sup>3</sup>, Face-Adapter<sup>4</sup>, and LivePortrait<sup>5</sup>.

For the IP-Adapter baseline, we use the implementation and pre-trained weights available on HuggingFace’s difusers library<sup>6</sup> in a dual-adapter SDXL configuration.<sup>7</sup> One “style” adapter is conditioned on a CLIP embedding of the source image, and the second “face” adapter—fine-tuned for facial representation tasks—is conditioned on a CLIP embedding of the driving image. The “style” and “face” adapters are assigned scale values of 0.5 and 0.6, respectively; these scale parameters control the strength of the corresponding image conditioning. The guidance scale is set to 3.0 to perform 27 denoising steps. In practice, we initialize the process with 30 denoising steps but skip the first three, and initialize the noisy latents with a noisy version of the source image, with the noise strength parameter set to 0.9. For classifier-free guidance, we drop the text prompt and use a noisy version of the driving image as the negative branch.

## A.2 Additional Results

Comparison against Text-driven Image Editing Approaches. Difusion models have gained significant attention for instruction-based image editing.

Ours

Source

InstructPix2Pix

LEDITS++  
![](images/965e27d9f23bd77a44aa7b489c0ee25659b4620603713ff12759b30a7e4fe2b4.jpg)  
Fig. 10: Comparing our facial replacement with instruction-based image editing approaches [7,8]. Note that, our approach uses a masked driving image (show in red inset), while the image editing approaches do not require a driving image. Our approach makes realistic edits following the driving image, while the image editing methods have no control over the facial edits, leading to unrealistic changes to personal identity.

SDEdit [34] proposes a training-free method for global, text-driven modifications but often fails to preserve regions outside the editing target. Subsequent approaches [1,38] introduce user-defined masks to enable localized control. Other methods [9, 36, 41] focus on manipulating attention maps, while LEDITS++ [7] integrates both attention and noise-based masking to achieve more precise edits. Fully supervised pipelines [8, 56, 68, 71, 72] support end-to-end editing from textual instructions. However, these models are not explicitly optimized for identity preservation in face-specific tasks. This limitation is evident in Fig. 10, where both InstructPix2Pix and LEDITS++ produce suboptimal results in terms of visual fidelity and identity consistency.

Quantitative Results. Tab. 3 provides a per-dataset breakdown of metrics for all benchmarks for replacement-ID setup. Please note that the FID scores for Face-Adapter [20] and HyperReenact [5] may be unreliable due to face detection failures during evaluation. Specifically, Face-Adapter fails on 4,325 (FFHQ), 4,813 (VoxCeleb ID), 68 (PicturePerfect), and 4,896 (VoxCeleb OOD) images out of 5,000. HyperReenact fails on 297 (FFHQ), 1,052 (VoxCeleb ID), 901 (PicturePerfect), and 1,264 (VoxCeleb OOD) images. These failures significantly impact the quality of FID estimation for these methods.

Source  
Driving  
Ours  
SeeDream 4.0  
![](images/075e781f3bac131f4d766857d76eb0f75fd8c4186fe12c358660c28fc397e5fe.jpg)  
Fig. 11: Comparing our facial replacement with multimodal-based editing approach [48]. Note that, our approach uses a masked driving image (showed in red inset), while the SeeDream4.0 uses the full driving image. Our approach makes realistic edits following the driving image, while the image editing method has quality output but has tendency to change person’s identity.

![](images/3a7eba729cc159267aeb31d989e61f3d9b9442be27abf5820dd0d6438d9d1c35.jpg)  
Source  
Target  
Output

Fig. 12: Although our proposed approach achieves more realistic facial replacement compared to prior methods, it still exhibits limitations in certain areas, particularly around the eyes and hair.  
Table 3: We quantitatively compare our method with prior approaches (± Standard Error of Mean) on replacement-ID testsets FFHQ [26] and VoxCeleb [11, 37].
<table><tr><td colspan="8">Replacement in-distribution (ID)</td></tr><tr><td colspan="8">FFHQ</td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td>DAM</td><td>0.557 ± 0.0034</td><td>25.28</td><td> $2 . 9 1 \pm 0 . 0 1 2 8$ </td><td> $0 . 2 7 9 6 \pm 0 . 0 0 2 2$ </td><td> $1 1 . 8 9 \pm 0 . 1 2 4 7$ </td><td> $\mathbf { 0 . 0 3 8 0 \ : \pm { \ : 0 . 0 0 0 3 } }$ </td><td> $\mathbf { 2 1 . 3 7 \ : \pm { \ : 0 . 0 4 9 1 } }$ </td></tr><tr><td>FADM</td><td>0.542 ± 0.0034</td><td>43.24</td><td>2.97 ± 0.0138</td><td> $0 . 2 8 4 9 \pm 0 . 0 0 2 2$ </td><td> $1 3 . 2 8 \pm 0 . 1 2 7 8$ </td><td> $0 . 0 4 5 3 \pm 0 . 0 0 0 4$ </td><td> $1 9 . 7 6 \pm 0 . 0 4 6 4$ </td></tr><tr><td>IP-Adapter</td><td> $0 . 3 1 2 \pm 0 . 0 0 1 8$ </td><td>17.22</td><td> $3 . 0 1 \pm 0 . 0 1 1 6$ </td><td> $0 . 2 9 1 0 \pm 0 . 0 0 2 2$ </td><td> $1 6 . 2 8 \pm 0 . 1 8 3 9$ </td><td> $0 . 0 8 3 5 \pm 0 . 0 0 0 4$ </td><td> $1 4 . 7 8 \pm 0 . 0 3 1 2$ </td></tr><tr><td>FaceAdapter</td><td> $0 . 4 2 9 \pm 0 . 0 0 6 4$ </td><td>122.47</td><td> $2 . 9 7 \pm 0 . 0 3 2 7$ </td><td> $\underline { { 0 . 2 4 3 9 \pm 0 . 0 0 4 9 } }$ </td><td> $1 0 . 3 7 \pm 0 . 3 3 8 3$ </td><td> $0 . 1 2 5 2 \pm 0 . 0 0 0 9$ </td><td> $1 0 . 3 2 \pm 0 . 0 8 2 4$ </td></tr><tr><td>HyperReenact</td><td> $0 . 4 3 1 \pm 0 . 0 0 1 9$ </td><td>111.89</td><td>2.79 ± 0.0118</td><td> $\mathbf { 0 . 2 4 3 9 \ : \pm { \ : 0 . 0 0 1 9 } }$ </td><td> $1 1 . 1 4 \pm 0 . 0 9 1 4$ </td><td> $0 . 1 1 5 2 \pm 0 . 0 0 0 4$ </td><td> $1 3 . 5 6 \pm 0 . 0 2 4 5$ </td></tr><tr><td>LivePortrait</td><td>0.676 ± 0.0031</td><td>11.18</td><td>2.79 ± 0.0113</td><td> $0 . 2 8 0 4 \pm 0 . 0 0 2 1$ </td><td> $\underline { { 8 . 4 4 \pm 0 . 0 9 0 3 } }$ </td><td> $\underline { { 0 . 0 4 3 2 \pm 0 . 0 0 0 3 } }$ </td><td> $\underline { { 2 0 . 2 3 \pm 0 . 0 5 8 4 } }$ </td></tr><tr><td>Ours</td><td>0.863 ± 0.0017</td><td>6.51</td><td>2.86 ± 0.0116</td><td> $0 . 2 7 9 7 \pm 0 . 0 0 2 1$ </td><td>5.35 ± 0.0647</td><td> $0 . 0 6 2 0 \pm 0 . 0 0 0 3$ </td><td>18.46 ± 0.0434</td></tr><tr><td colspan="8">VoxCeleb</td></tr><tr><td>DAM</td><td>0.466 ± 0.0031</td><td>24.43</td><td>2.89 ± 0.0138</td><td>0.2412 ± 0.0017</td><td>13.01 ± 0.1220</td><td>0.0345 ± 0.0003</td><td>21.96 ± 0.0433</td></tr><tr><td>FADM</td><td>0.413 ± 0.0032</td><td>53.60</td><td>2.99 ± 0.0163</td><td> $0 . 2 4 3 7 \pm 0 . 0 0 1 7$ </td><td>13.55 ± 0.1268</td><td>0.0457 ± 0.0003</td><td>20.43 ± 0.0436</td></tr><tr><td>IP-Adapter</td><td>0.253 ± 0.0018</td><td>21.33</td><td>3.12 ± 0.0133</td><td> $0 . 2 6 9 8 \pm 0 . 0 0 1 9$ </td><td>20.10 ± 0.2033</td><td> $0 . 0 9 3 8 \pm 0 . 0 0 0 3$ </td><td> $1 6 . 0 3 \pm 0 . 0 2 7 9$ </td></tr><tr><td>FaceAdapter</td><td> $0 . 3 0 9 \pm 0 . 0 1 2 6$ </td><td>140.57</td><td> $3 . 1 4 \pm 0 . 0 6 5 8$ </td><td> $0 . 2 8 3 3 \pm 0 . 0 0 8 6$ </td><td> $1 2 . 9 0 \pm 0 . 7 4 2 5$ </td><td> $0 . 1 0 0 3 \pm 0 . 0 0 2 7$ </td><td> $1 0 . 5 4 \pm 0 . 1 6 6 6$ </td></tr><tr><td>HyperReenact</td><td> $0 . 3 7 9 \pm 0 . 0 0 2 2$ </td><td>100.97</td><td> $3 . 2 9 \pm 0 . 0 1 4 1$ </td><td> $0 . 3 1 4 7 \pm 0 . 0 0 1 8$ </td><td> $1 1 . 2 9 \pm 0 . 1 0 3 4$ </td><td> $0 . 0 9 2 5 \pm 0 . 0 0 0 4$ </td><td> $1 3 . 8 2 \pm 0 . 0 2 5 4$ </td></tr><tr><td>LivePortrait</td><td> $\underline { { 0 . 4 8 6 \pm 0 . 0 0 2 9 } }$ </td><td>13.57</td><td> $\mathbf { 2 . 7 2 \ : \pm { \ : 0 . 0 1 1 6 } }$ </td><td> $0 . 2 4 5 5 \pm 0 . 0 0 1 7$ </td><td> $\underline { { 1 0 . 3 7 \pm 0 . 1 0 3 7 } }$ </td><td> $\mathbf { 0 . 0 2 9 0 \ : \pm { \ : 0 . 0 0 0 2 } }$ </td><td> $2 0 . 7 8 \pm 0 . 0 5 1 7$ </td></tr><tr><td>Ours</td><td> $\mathbf { 0 . 7 7 2 \ : \pm { \ : 0 . 0 0 2 7 } }$ </td><td>9.12</td><td> $\underline { { 2 . 7 6 \pm 0 . 0 1 2 2 } }$ </td><td> $\mathbf { 0 . 2 3 8 8 \ : \pm { \ : 0 . 0 0 1 7 } }$ </td><td>6.55 ± 0.0762</td><td> $0 . 0 6 6 0 \pm 0 . 0 0 0 3$ </td><td> $1 9 . 8 9 \pm 0 . 0 3 8 9$ </td></tr></table>

Issues of Image Quality Metrics. In Fig. 13, we highlight the discrepancy between standard image quality metrics and human judgment. Although our results exhibit superior visual quality compared to baseline methods—as further supported by our user study—the quantitative metrics tend to undervalue our approach.

![](images/7030dbf3f09bb351d945589adf8bb5868a965d64125bcee6b46ad4e1adc77715.jpg)  
Fig. 13: Metrics like PSNR and LPIPS often favor blurry, low-quality reconstructions due to sensitivity to slight misalignments, failing to align with human judgment.

![](images/6b786dff7d1731a15c80626291a56ed86f38ec43ac4fa3e76f1978867a130e32.jpg)  
Fig. 14: Ablation: Robustness to data augmentations. Our solution demonstrates the ability to mitigate the efects of diferent data augmentations, resulting in more stable and visually appealing results.

Model Robustness to Various Transformations. In Fig. 14, we assess our model’s robustness to various augmentations applied to the face in the driving image, including color jittering, translation, rotation, and scaling. The results indicate that our model is generally consistent despite large variations in facial color, position, orientation, and size. However, it exhibits dificulty in preserving identity when the face in the driving image is either extremely small (first row of Scaling driving) or excessively large (fourth row of Scaling driving). This limitation may stem from information loss during the image-to-latent conversion process—where small faces lack suficient detail.

Source  
Target  
![](images/01e0d24f81ab47aeed775e96ea448e9cc8de6898848f40b67b7ad5dfeebce026.jpg)  
Ours

Source  
Target  
![](images/c5d23512a66b316f08c16396da954dbf1e0ff268fe69402bee3c6ecc45ea9255.jpg)  
Ours  
Fig. 15: Additional face replacement (in-distribution) results

Additional Visual Results. Fig. 15 and Fig. 16 present additional visual results of our proposed method for the face replacement task. Furthermore, Fig. 11 showcases direct comparisons between our method and SeeDream4.0 [48]. As can be seen, SeeDream4.0 yields promising results but has a tendency to alter the identity of the source image, making it unreliable. We also don’t include this approach in our quantitative study due to the fact that inference using the See-Dream4.0 model was extremely slow and required manual intervention.

Additional Ablation Results. Tab. 4 Interestingly, the No Relighting, No Augmentation, and Driving as-is variants often achieve higher reconstructionoriented metrics than our final model. However, these variants simplify the task by either reducing appearance variation during training or exposing richer targetrelated appearance information to the model. In particular, the Driving as-is variant can preserve facial appearance cues from the driving image, leading to shortcut solutions where facial details are partially copied into the generated output. Likewise, removing relighting or data augmentation weakens regulariza tion and reduces the need for robust identity disentanglement. As a result, these variants can produce outputs that are numerically closer to the ground-truth image, yielding improved CSIM, LPIPS, FID, and PSNR scores, while relying on appearance leakage rather than genuine identity-preserving synthesis. We therefore retain relighting and augmentation in the final model, as they encourage stronger disentanglement and better generalization under realistic deployment conditions.

Source  
Target  
Ours  
![](images/29a3a5455d21b51e7faa6ebcb015d4bbdc9eb6e28cf40ec4c71f6170c5f8ef38.jpg)

Source  
Target  
Ours  
![](images/42b53698419405e72fdf757bbc635433fb19c0f0b16445259f23df4c9c3bab9c.jpg)  
Fig. 16: Additional face replacement (out-of-distribution) results

Failure Cases. Fig. 12 illustrates representative failure cases in the sameidentity face replacement task. Most failures are caused by suboptimal generation of intricate details such as eyes and hair, and this is caused by the limitations of the backbone difusion [42] architecture.

Table 4: Additional ablation study of our identity-preserving face replacement framework. We compare the final model against variants without relighting, without data augmentation, and using the driving image as-is. While some ablations achieve higher reconstruction-oriented metrics, these settings introduce shortcut cues that can artificially increase similarity to the target image and therefore do not reflect robust identity-preserving face replacement.
<table><tr><td colspan="8">Replacement in-distribution (ID)</td></tr><tr><td colspan="8">FFHQ + VoxCeleb2 (LivePortrait)</td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>LPIPS↓</td><td>PSNR↑</td></tr><tr><td>Ours</td><td>0.818</td><td>7.81</td><td>2.81</td><td>0.2592</td><td>5.95</td><td>0.0640</td><td>19.17</td></tr><tr><td>Channel-wise</td><td>0.689</td><td>55.19</td><td>3.67</td><td>0.3542</td><td>6.18</td><td>0.0831</td><td>16.11</td></tr><tr><td>SDv1.5</td><td>0.776</td><td>9.90</td><td>2.87</td><td>0.2620</td><td>6.58</td><td>0.0564</td><td>17.55</td></tr><tr><td>No Relighting</td><td>0.836</td><td>7.49</td><td>2.81</td><td>0.2632</td><td>5.88</td><td>0.0569</td><td>18.68</td></tr><tr><td>No Augmentation</td><td>0.912</td><td>6.73</td><td>2.80</td><td>0.2668</td><td>4.35</td><td>0.0202</td><td>22.66</td></tr><tr><td>Driving as-is</td><td>0.945</td><td>4.03</td><td>2.80</td><td>0.2656</td><td>4.34</td><td>0.0045</td><td>33.72</td></tr></table>

<table><tr><td colspan="10">Replacement out-of-distribution (OOD)</td></tr><tr><td></td><td></td><td>PicturePerfect (Real)</td><td></td><td></td><td></td><td></td><td>VoxCeleb2 (LivePortrait)</td><td></td><td></td><td></td></tr><tr><td>Methods</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td><td>CSIM↑</td><td>FID↓</td><td>AED↓</td><td>APD↓</td><td>AGD↓</td></tr><tr><td>Ours</td><td>0.631</td><td>28.08</td><td>2.50</td><td>0.2262</td><td>9.87</td><td>0.579</td><td>12.44</td><td>2.78</td><td>0.2366</td><td>12.78</td></tr><tr><td>Channel-wise</td><td>0.618</td><td>80.70</td><td>3.37</td><td>0.3113</td><td>8.47</td><td>0.548</td><td>62.26</td><td>3.69</td><td>0.3418</td><td>11.80</td></tr><tr><td>SDv1.5</td><td>0.655</td><td>30.98</td><td>2.50</td><td>0.2164</td><td>9.71</td><td>0.576</td><td>12.96</td><td>2.84</td><td>0.2413</td><td>13.19</td></tr><tr><td>No Relighting</td><td>0.759</td><td>26.92</td><td>2.43</td><td>0.2116</td><td>8.67</td><td>0.685</td><td>11.91</td><td>2.83</td><td>0.2465</td><td>13.00</td></tr><tr><td>No Augmentation</td><td>0.720</td><td>28.90</td><td>2.38</td><td>0.2139</td><td>6.99</td><td>0.737</td><td>12.63</td><td>2.74</td><td>0.2515</td><td>7.29</td></tr><tr><td>Driving as-is</td><td>0.713</td><td>21.12</td><td>2.38</td><td>0.2119</td><td>7.48</td><td>0.772</td><td>12.28</td><td>2.78</td><td>0.2523</td><td>7.21</td></tr></table>

Source  
Target  
Driving  
![](images/1a7d0c921de0cf4f489656659a213481ef1bd8cae441bfd9a4f81f40a01ef4f2.jpg)  
Output  
Fig. 17: Experiment with cross-identity facial replacement where the identity along with the expression from the driving image gets transferred to source.

Source  
Target  
Driving  
Output  
![](images/804e38c8b634d8f689018ef2b85a519cc7ec0623d6a3624d695999cd0d5994db.jpg)  
Fig. 18: Experiment with cross-identity facial replacement where only the expression from the driving image gets transferred to source while keeping the identity unaltered.

## B PicturePerfect Dataset

Most existing reenactment datasets are biased toward interview-style YouTube videos, where subjects are primarily talking with limited head movement and facial variation. In contrast, we introduce the PicturePerfect dataset, a synthetic dataset featuring diverse head poses and expressions, generated using a portrait animation pipeline. Our dataset comprises approximately 500,000 images spanning over 6,724 unique identities, and from which more than 10 million distinct source–target pairs can be sampled. Sample images from our dataset are shown in Fig. 19. Each face in our dataset is available under 21 distinct poses and expressions, including a neutral image and 20 expressive variants generated using LivePortrait [19], a face animation model.

![](images/50727c586d8160ff811f7798824f74ff033d8c191de98000996131ac5d093040.jpg)  
Fig. 19: PicturePerfect dataset samples. For each neutral face (top row), we generate 20 expressive variants using a face animation model [19]. For visualization purposes, we display four randomly sampled expressive variants per neutral face.

To construct the dataset, we leverage an internal face video dataset and LivePortrait [19], a model capable of animating a face based on a driving video. An overview of the animation pipeline is shown in Fig. 20.

![](images/12696e627e768fba6892f5ececa2fdf1c4970639c9546c210bca75e11d751a7e.jpg)  
Fig. 20: PicturePerfect Dataset generation process overview: Neutral looking faces are detected and extracted from source videos, which are guided via driving videos to generate expressive animated faces using LivePortrait [19].

![](images/c222a0f6571d642f6b5a6021020c6b99b18e629230192f18f3b168ea1a5179db.jpg)  
Fig. 21: Overview of some example driving videos used to animate the neutral faces with LivePortrait [19].

## B.1 Video Crowdsourcing

To collect videos, we used a custom platform in collaboration with crowdsourcing providers to recruit participants from diverse gender, geographic, and ethnic backgrounds. Data was collected under direct agreements with crowd workers, permitting both research and commercial use and ensuring compliance with privacy regulations, including GDPR-equivalent standards. All videos were manually reviewed to exclude issues such as individuals in the background. Personally identifiable information and metadata were removed to ensure participant privacy. All contributors were fairly and appropriately compensated according to regional standards and signed consent forms explicitly permitting research and commercial use of their video and audio data, including training AI models. A contact email will be provided on the dataset release page for participants to request data removal at any time. Participants were instructed to record short videos while performing specific actions and expressions, such as “smiling gently at the camera while sitting in front of a computer”, “pretending to be shocked while looking at the screen”, “looking down with a sad expression while sitting at a desk” and so on. Each participant was encouraged to submit multiple videos featuring varied backgrounds, outfits, and lighting conditions.

![](images/f04760e3bc6eb584d0aebd0069b8832459a876020989236c4a85b884d786443e.jpg)  
Fig. 22: Random samples from the dataset. The PicturePerfect dataset features a wide diversity of backgrounds, lighting conditions, ethnicities, clothing styles, poses, and facial expressions.

## B.2 Data Pre-processing and Filtering

From the collected videos, we filter and retain only those that fall into the following predefined categories of facial expressions and head gestures:

– making eye contact with the camera while sitting in front of a computer

– making eye contact with the camera while sitting at a distance

– making eye contact with the camera while standing

no facial expression

no gesture

From the filtered videos, we randomly sample pairs of frames from segments where the subject is looking directly at the camera. To ensure the selection of truly neutral expressions, we extract facial landmarks and bounding boxes using FaceNet [17]. We then compute the angles between the left and right eyes and the nose, as well as left and right corners of the mouth and the nose, ensuring the head orientation is within an empirically determined threshold of $1 0 ^ { \circ }$ . To further eliminate head tilt, we also check that the vertical angle between the nose and the mid-point of both the eyes is below 10<sup>◦</sup>.

Images that pass both filters are standardized by cropping around the midpoint of the detected bounding box, using a bounding box scaled by a factor of 2. This process yields approximately 25,000 neutral face images across 6,724 unique identities.

To enable neutral-to-any-expression animation, we manually select 200 diverse driving videos. In Fig. 21, we provide some example driving videos. For each neutral face, we randomly sample 20 unique driving videos and animate the source using LivePortrait [19]. An overview of the animation pipeline using LivePortrait is shown in Fig. 20. From each resulting video, we extract the final frame as the animated output. This results in 20 animated images per source, culminating in a dataset of approximately 500,000 images comprising both neutral and expressive faces.

We provide an overview of the images in Fig. 22 to demonstrate the data diversity for backgrounds, lighting, gender, ethnicity, head poses, facial expression.

## B.3 Caption Generation

To generate descriptive captions for the upscaled images, we use LLaVA-1.5- 7B [32] with the instruction: “Describe this real-world image and its style as a long detailed caption. Include details about the person’s facial expression and describe face attributes. Ignore the background and clothing information for the caption.”