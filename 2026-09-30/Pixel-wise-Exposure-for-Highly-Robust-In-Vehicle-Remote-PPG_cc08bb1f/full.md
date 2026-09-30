# Pixel-wise Exposure for Highly Robust In-Vehicle Remote-PPG

Jieying Wang, Member, IEEE, Xinqi Cai, Caifeng Shan\*, Senior Member, IEEE, and Wenjin Wang\*

Abstract—Remote photoplethysmography (rPPG) offers a promising non-contact solution for heart rate monitoring, yet its real-world robustness is fundamentally limited by an inherent hardware limitation: existing camera exposure control paradigms—whether fixed, or auto-exposure—impose a uniform exposure time across all pixels within a frame. In high-dynamicrange scenes such as automotive cabins with strong directional sunlight, this spatially invariant exposure constraint inevitably leads to localized facial overexposure or underexposure, irreversibly corrupting the subtle pulsatile signals essential for rPPG at the point of capture—a physical degradation that no downstream algorithm can recover. To overcome this bottleneck, we propose PixExpo (Pixel-wise Exposure), a “temporal-for-spatial” framework that sequentially captures frames under a predefined cyclic exposure schedule and performs non-iterative pixel-wise fusion. At each pixel location, PixExpo selects the observation closest to an rPPG-motivated target intensity. This criterion seeks to reduce local saturation and severe underexposure rather than optimize perceptual appearance. PixExpo requires no sensor modification but assumes programmable frame-level exposure control. We validate the proposed PixExpo framework using our newly introduced MEX-Drive dataset, comprising 48 participants under real-world driving conditions. Experimental results demonstrate that PixExpo outperforms manufacturedefault auto-exposure methods, reducing the mean absolute error (MAE) by 7.21 bpm (from 13.94 to 6.73 bpm) and increasing the success rate by 37.29 percentage points (from 25.95% to 63.24%) across challenging driving scenarios.

Index Terms—Remote photoplethysmography, Multi-exposure fusion, Camera exposure control, Driver monitoring.

## I. INTRODUCTION

Remote photoplethysmography (rPPG) estimates heart rate (HR) from subtle pulse-induced skin-color changes captured by a camera, enabling contactless monitoring in telemedicine, fitness tracking, and driver monitoring [1]–[3]. However, its real-world performance remains highly sensitive to illumination [4]. This challenge is particularly pronounced in vehicle cabins, where rapid lighting transitions and high-contrast shadows frequently occur, as shown in Fig. 1 (a).

Critically, the vulnerability of rPPG under such conditions is not merely an algorithmic issue; it stems from a more fundamental, hardware-level limitation inherent to virtually all existing camera exposure control paradigms. Whether fixed, auto-exposure (AE), or adaptive [5], every mainstream method imposes a single exposure time uniformly across all pixels within a frame. Under unbalanced illumination with extreme local contrast (e.g., side sunlight, shadow occlusion, and sunset glare in driving scenes), this one-exposure-fits-all mechanism makes it physically impossible to maintain optimal luminance for each skin pixel simultaneously. Inevitably, some pixels fall into overexposure (signal clipping) while others suffer from underexposure (noise dominance), resulting in irreversible corruption of the weak physiological signal. Most importantly, this imaging-source signal loss is a pure hardware/physical limitation that cannot be compensated, recovered, or repaired by any rPPG extraction or enhancement algorithms—once the photon-level information is lost due to sensor saturation, all subsequent post-processing efforts are futile. Thus, the conventional global exposure paradigm constitutes a fundamental, previously overlooked bottleneck that places a hard upper bound on the robustness of any rPPG system in real-world environments.

Existing approaches have two further limitations: (1) mainstream exposure adjustment relies on iterative fitting [6] or feedback control [7], introducing non-negligible latency that fails to track high-frequency rapid lighting transitions in driving; (2) nearly all exposure optimization and multi-exposure fusion (MEF) criteria [8]–[10] are designed for human visual perception, which disrupt the subtle yet critical inter-frame chrominance variations essential for rPPG monitoring.

To address these limitations, we propose PixExpo, a computational framework that uses a “temporal-for-spatial” strategy to achieve pixel-wise exposure adaptation without modifying the camera’s sensor architecture. Specifically, PixExpo uses a conventional camera to rapidly capture a cyclic sequence of K frames at different global exposure settings and synthesizes a fused frame by selecting the most suitable RGB observation at each pixel location, as illustrated in Fig. 1(b). Thus, each captured frame uses a single global exposure time, while different pixels in the fused frame can originate from different exposure settings. This non-iterative process reduces the risk of localized saturation and severe underexposure while avoiding response delays from closed-loop exposure search. Unlike conventional MEF, PixExpo uses an rPPG-motivated photometric criterion rather than a perceptual-quality objective. Notably, this framework requires no customized sensor or hardware modification, but relies on a programmable interface that supports disabling automatic exposure, reliable frame-level exposure updates, and correct association between each frame and its applied exposure setting. Such deterministic control is commonly available in industrial cameras. Consumer-grade cameras remain applicable when their vendor APIs provide these access to the full control of camera parameters programmably and a sufficiently high acquisition frame rate.

![](images/45f53e3b684ce7cf75f7d6926cc83464c1435c1ed2b4c7f35f6722db39ec2a3e.jpg)  
Fig. 1: Overview of illumination challenges for in-vehicle rPPG and the proposed PixExpo framework. (a) Representative high-dynamic-range (HDR) driving scenarios from the MEX-Drive dataset. (b) Multi-exposure acquisition and pixel-wise fusion pipeline of PixExpo.

We summarize our contributions as follows:

• We identify the spatially invariant exposure constraint as a fundamental yet previously overlooked hardware-level bottleneck for rPPG robustness, and propose PixExpo, a “temporal-for-spatial” acquisition paradigm that breaks this physical limitation via a predefined cyclic multiexposure schedule and pixel-wise fusion, ensuring robust signal preservation in fast-varying lighting environments without iterative algorithmic latency.

• We release MEX-Drive<sup>1</sup>, a new dataset comprising synchronized six-level multi-exposure facial videos alongside clinical-grade ECG data from 48 participants, facilitating further research.

## II. RELATED WORK

Remote PPG has been extensively studied for non-contact physiological monitoring, yet its deployment under challenging real-world illumination remains an open problem. While prior work has reviewed in-vehicle rPPG monitoring [3], [11]– [13], this section focuses specifically on exposure control strategies for robust physiological signal extraction. We organize representative methods along three dimensions: (i) spatial robustness—avoiding localized saturation or underexposure in the facial region; (ii) real-time adaptability—responding to rapid lighting changes without algorithmic latency; and (iii) rPPG-motivated design—whether the criterion targets physiological measurement rather than perceptual image quality. Table I provides a structured overview.

## A. Standard Auto-Exposure Algorithms

Conventional auto-exposure (AE) methods estimate a single exposure setting from image statistics. Early approaches used numerical root-finding methods, such as false-position and modified secant algorithms, to exploit the monotonic relationship between exposure and image brightness [14], [15]. Backlight-aware methods adjust the target luminance using brightness statistics [16], while hardware-oriented implementations classify exposure states by counting bright and dark pixels to reduce control latency [17].

For robotic and automotive vision, exposure objectives have been extended beyond global brightness. Shim et al. [18] introduced gradient-based feedback control for individual cameras and brightness balancing across multiple cameras. Other methods combine image gradients and entropy for feature tracking [19], incorporate photometric calibration and motionblur constraints [20], or use Bayesian optimization to balance gradients, saturation, noise, and signal-to-noise ratio [6]. These methods improve convergence, visibility, or machineperception performance, but still produce one exposure setting per camera at each instant. Consequently, they cannot independently correct sunlit and shadowed facial regions within the same frame. Their responsiveness is also method-dependent: low-latency implementations are possible, whereas iterative optimization or feedback updates may lag abrupt illumination changes.

## B. Multi-Exposure Image and Video Processing

Multi-exposure fusion (MEF) combines complementary observations captured at different exposure levels to alleviate the spatial limitations of global exposure [8], [9]. Image-based methods typically fuse an exposure bracket into a single image using spatial-domain [21]–[25], transform-domain [26], [27], or learning-based techniques [29], [30].

Exposure diversity has also been explored for video enhancement. Shen et al. [31] combined cyclic short–long exposures with optical-flow-based alignment, restoration, and interpolation to reconstruct sharp, noise-reduced, high-framerate videos. These image and video methods primarily target visual quality, which does not necessarily imply preservation of pulsation-induced temporal color variations.

TABLE I: Comparison of Existing Exposure Control Strategies and the Proposed PixExpo
<table><tr><td>Method Category</td><td>Representative Works</td><td>Spatial Robustness (Avoids Local Saturation)</td><td>Real-time Adaptability (Low Latency)</td><td>rPPG-Oriented</td></tr><tr><td>Standard Auto-Exposure</td><td>[6], [14]–[20]</td><td>x (Global average)</td><td>~ (Partially)</td><td>x (Vision-oriented)</td></tr><tr><td>Traditional MEF</td><td>[21]-[31]</td><td>√ (Pixel/Region-level)</td><td>2 (Partially)</td><td>x (Vision-oriented)</td></tr><tr><td>rPPG-specific Global AE</td><td>[5], [7], [32]–[35]</td><td>x (Global average)</td><td>~ (Partially)</td><td>√ (rPPG-oriented)</td></tr><tr><td>Proposed framework</td><td>PixExpo (Ours)</td><td>√ (Pixel-wise fusion)</td><td>√ (No iterative search; sequential capture)</td><td>√ (rPPG-oriented)</td></tr></table>

Note: ✓, ✗, and ∼ denote “explicitly addressed”, “not explicitly addressed”, and “partially addressed”, respectively.

## C. rPPG-Specific Global Auto-Exposure

Recognizing the unsuitability of vision-oriented methods, recent studies have proposed exposure control strategies explicitly optimized for rPPG signal fidelity [5], [7], [32]– [35]. These methods incorporate physiological awareness into the control loop, for instance by maximizing the signal-tonoise ratio within the sensor’s linear dynamic range [5], utilizing signal quality metrics as feedback [32], or employing proportional-integral-derivative (PID) controllers to stabilize facial luminance against ambient fluctuations [7], [33], [34]. A real-time triplet-frame linear fitting approach was also proposed in [35] to dynamically adjust global exposure in vehicular environments. While these approaches successfully shift the optimization target to be rPPG-motivated and partially mitigate latency through faster algorithms [35], they remain fundamentally trapped in the global exposure paradigm. They compute a single exposure time intended to optimize the average condition across the entire facial ROI. Consequently, they offer no improvement in spatial robustness—localized saturation on a sunlit cheek or underexposure on a shadowed brow remains an inevitable consequence when the intra-scene dynamic range exceeds the sensor’s native capacity.

## D. Summary and Positioning of PixExpo

As summarized in Table I, conventional AE provides global exposure control with method-dependent latency, traditional MEF handles spatial exposure variation but is primarily perception-oriented, and rPPG-specific AE remains spatially global. PixExpo combines a predefined cyclic exposure schedule with rPPG-motivated pixel-wise fusion. The predefined schedule avoids iterative exposure search, while pixel-wise selection explicitly addresses local exposure variation. Crucially, unlike traditional MEF, the fusion strategy in PixExpo is rPPGmotivated, ensuring that the physiological integrity of the skin pixels is prioritized over human visual aesthetics.

## III. METHODOLOGY

PixExpo is a computational pixel-wise multi-exposure framework designed for spatially nonuniform in-vehicle illumination. It approximates a spatial exposure map by sequentially acquiring globally exposed frames and selecting or weighting the observations at each pixel location.

## A. Theoretical Foundation

To describe the acquisition-level limitation of global exposure control, we use a simplified image-formation model. The digital intensity at spatial location x and time $t ,$ captured with exposure time E, is expressed as

$$
I ( x , t ; E ) = \mathcal { Q } \left( c E L ( x , t ) \left[ R _ { \mathrm { D C } } ( x ) + R _ { \mathrm { A C } } ( x , t ) \right] + N ( x , t ) \right) ,\tag{1}
$$

where Q(·) maps the sensor response to the processed digital intensity range, c is a system response constant, $L ( x , t )$ is the incident illumination, and $N ( x , t )$ represents sensor and quantization noise. The skin reflectance consists of a stationary component $R _ { \mathrm { D C } } ( x )$ and a small pulse-induced component $R _ { \mathrm { { A C } } } ( x , t )$ . In the 8-bit RGB representation used for fusion in this study, the processed intensity range is 0–255; this does not refer to the native ADC precision of the image sensor.

Within the approximately linear and unsaturated response range, the magnitude of the pulse-related intensity variation is proportional to $E L ( x , t ) R _ { \mathrm { A C } } ( x , t )$ . When the pixel value is clipped at the upper limit, part or all of this temporal variation is lost. At very low exposure, the pulsatile variation may become comparable to the sensor and quantization noise, thereby reducing rPPG SNR.

Let $I _ { \mathrm { D C } } ( x , t ; E )$ denote the local DC intensity under exposure $E ,$ and let $\Omega _ { \mathrm { R O I } }$ denote the facial region of interest. The set of photometrically valid pixels is defined as

$$
\Omega _ { \mathrm { v a l i d } } ( E , t ) = \left\{ x \in \Omega _ { \mathrm { R O I } } \mid I _ { \mathrm { m i n } } \leq I _ { \mathrm { D C } } ( x , t ; E ) \leq I _ { \mathrm { m a x } } \right\} ,\tag{2}
$$

where $[ I _ { \mathrm { m i n } } , I _ { \mathrm { m a x } } ]$ denotes the usable intensity range. When the effective illumination range across the face exceeds the usable dynamic range of the sensor, no single scalar exposure E can place all facial locations within this range. Consequently, a global exposure setting reduces the number of skin pixels that provide photometrically suitable observations for rPPG.

An ideal spatial exposure map would select, for each location, the exposure that brings its DC intensity closest to a target operating point:

$$
E ^ { * } ( x , t ) = \arg \operatorname* { m i n } _ { E } \left| I _ { \mathrm { D C } } ( x , t ; E ) - I _ { \mathrm { t a r g e t } } \right| .\tag{3}
$$

Because the camera used in this study cannot assign independent exposure times to individual pixels, PixExpo approximates this ideal map through sequential global exposures and computational pixel-wise fusion.

## B. The Proposed PixExpo Framework

PixExpo consists of two modules: SDK-based multiexposure acquisition and pixel-wise multi-exposure fusion. The acquisition module repeatedly captures frames using a predefined exposure schedule, while the fusion module evaluates and combines the corresponding pixel observations.

1) SDK-Based Multi-Exposure Acquisition: Automatic exposure is disabled, and the camera SDK updates the exposure time on a frame-by-frame basis according to the predefined cyclic schedule

$$
\begin{array} { r } { \xi = \{ E _ { 1 } , E _ { 2 } , . . . , E _ { N } \} , \qquad E _ { 1 } < E _ { 2 } < \cdot \cdot \cdot < E _ { N } , } \end{array}\tag{4}
$$

where $N = 6$ in the main configuration. Each acquisition cycle contains N consecutive frames, with the k-th frame captured using exposure time $E _ { k }$ . The schedule repeats continuously without iterative exposure estimation or feedback search.

The resulting sequence samples a range of facial brightness levels within each output cycle and provides the candidate observations for subsequent fusion. Because these frames are acquired sequentially rather than simultaneously, inter-frame timing and spatial misalignment are considered in the later analysis.

2) Pixel-Wise Multi-Exposure Fusion: For the k-th exposure frame, let $I _ { k } ( i , j )$ denote the mean RGB intensity of the pixel at the coordinate $( i , j )$ in the k-th sub-stream. The deviation $D _ { k } ( x , y )$ of the k-th exposure from this target is calculated as:

$$
D _ { k } ( i , j ) = | I _ { k } ( i , j ) - I _ { t a r g e t } | .\tag{5}
$$

The deviation $D _ { k } ( i , j )$ is an rPPG-motivated photometric proxy rather than a direct measure of physiological signal quality. It does not directly measure pulsatile amplitude, periodicity, chrominance variation, or rPPG SNR. Instead, it favors observations that are less likely to be affected by saturation or severe underexposure, both of which can suppress or obscure the weak pulsatile component during acquisition. In this study, $I _ { \mathrm { t a r g e t } }$ is empirically fixed at 140 on the processed 8-bit intensity scale for all recordings. This intermediate operating point provides margins against both saturation and severe underexposure.

At each pixel location, the exposure candidates are ranked according to their deviations:

$$
\begin{array} { c } { { \displaystyle { \bf S } ( i , j ) = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { N } \} , \quad \mathrm { s . t . } } } \\ { { { \cal D } _ { s _ { 1 } } ( i , j ) \le { \cal D } _ { s _ { 2 } } ( i , j ) \le \cdots \le { \cal D } _ { s _ { n } } ( i , j ) , } } \end{array}\tag{6}
$$

where $s _ { k } ( i , j )$ denotes the original exposure-channel index of the k-th ranked candidate. Thus, $s _ { 1 } ( i , j )$ identifies the exposure observation closest to $I _ { \mathrm { t a r g e t } }$

The fused RGB pixel is computed as

$$
\mathbf { I } _ { \mathrm { f u s e d } } ( i , j ) = \sum _ { k = 1 } ^ { N } w _ { k } ( i , j ) \mathbf { I } _ { s _ { k } } ( i , j ) , \qquad \sum _ { k = 1 } ^ { N } w _ { k } ( i , j ) = 1 .\tag{7}
$$

We consider three weighting strategies.

1) One-hot weighting:

$$
w _ { 1 } ( i , j ) = 1 , \qquad w _ { k } ( i , j ) = 0 , \quad k = 2 , \ldots , N .\tag{8}
$$

Only the highest-ranked candidate is retained, and its complete RGB vector is copied to the fused frame.

2) Gaussian decay weighting:

$$
w _ { k } ( i , j ) = \frac { \exp \left[ - ( k - 1 ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) \right] } { \sum _ { m = 1 } ^ { N } \exp \left[ - ( m - 1 ) ^ { 2 } / ( 2 \sigma ^ { 2 } ) \right] } .\tag{9}
$$

3) Dynamic inverse weighting:

$$
w _ { k } ( i , j ) = \frac { ( D _ { s _ { k } } ( i , j ) + \epsilon ) ^ { - 1 } } { \sum _ { m = 1 } ^ { N } \left( D _ { s _ { m } } ( i , j ) + \epsilon \right) ^ { - 1 } } ,\tag{10}
$$

where ϵ prevents division by zero. In the Gaussian strategy, k denotes the rank after sorting by photometric deviation, rather than the original exposure-channel index.

In this study, one-hot weighting is adopted as the default strategy. Unlike Gaussian decay and inverse weighting, which blend multiple exposure candidates to improve spatial smoothness, one-hot weighting preserves the complete RGB value of the candidate closest to the target intensity, thereby avoiding contributions from suboptimal exposures. This design choice is quantitatively evaluated in Sec. V-C.

## C. Potential Artifacts of Discrete Exposure Selection

One-hot selection can introduce spatial discontinuities at boundaries where neighboring pixels select different exposure levels. It may also introduce temporal variation when the selected exposure channel changes across successive output frames. Static differences in regional DC intensity can be partly reduced by temporal normalization, while spatial aggregation over the facial ROI can attenuate localized discontinuities. However, temporal band-pass filtering cannot remove exposure-switching components that fall within the heart-rate frequency band.

The influence of these artifacts must therefore be evaluated empirically rather than assumed to be negligible. Secs. V-C and V-D analyze this issue through the weighting-strategy ablation, inter-frame registration comparison, and spectral analysis of the frame switching ratio and mean selection index. PixExpo thus accepts possible visual discontinuities as a tradeoff of discrete exposure selection, while its effect on rPPG measurement is assessed using physiological metrics.

## IV. EXPERIMENT SETUP

## A. Dataset: MEX-Drive

MEX-Drive was collected to evaluate multi-exposure acquisition and fusion under spatially nonuniform in-vehicle illumination. Although it used the same hardware configuration and participant pool as the ExpDrive dataset [32], the recordings analyzed in this study were acquired using a distinct multi-exposure protocol focused on high-dynamicrange driving scenes, as shown in Fig. 2(b).

1) Experimental Design and Protocol: The study enrolled 48 licensed drivers, including 33 males and 15 females, aged 20–65 years (35.0 ± 12.4 years). All participants were Asian, with skin tones ranging from levels 4 to 6 on the 10-level Monk Skin Tone (MST) scale. Note that skin-tone variability was not a focus of this study, as PixExpo was designed to address dynamic illumination during driving and it does not include a mechanism specifically addressing skintone variation challenge in rPPG measurement. The protocol covered different illumination and weather conditions (sunny, rainy/overcast, and sunset glare), road types (highway and urban roads), three route segments (A–C), and natural driver activities such as yawning, talking, and checking the mirrors. The mean reference HR was 76.2 ± 10.8 bpm, as shown in Fig. 2(c). All participants received an explanation of the study and provided written informed consent. The protocol was approved by the Institutional Review Board of Southern University of Science and Technology (IRB No. 20240150).

(a) Data collection deployment  
![](images/819d7e1c2a28b9a1ff56b6ce3c2316773e444bc4ce68ade86757e073b6a463f4.jpg)

![](images/44600894bbda4c8be722c0fc67ba04427cb896d4a4411e7c773ac4912d68736d.jpg)  
(c) Age and HR distribution

![](images/113b4ede284f214b13a9ba92fff6c7e7584b989396b498abaa33e7cecb8408c0.jpg)

![](images/2801f5d593030fd8e789339209d10ad7eff073db12170fe9d86fd4b5a2d2c50a.jpg)  
Fig. 2: Experimental setup and characteristics of MEX-Drive. (a) Dual-camera deployment and reference ECG acquisition; (b) Representative high-dynamic-range driving conditions; (c) Age and reference-HR distributions of the 48 participants.

2) Acquisition Hardware: Two synchronized industrial RGB cameras (IDS UI-3160CP-C-HQ) were controlled through the IDS uEye API using a custom C++ acquisition program. Each 2 × 2 block of the 960 × 600 RGGB Bayer frames was converted into one RGB pixel, yielding a resolution of 480×300 pixels. For the cyclic multi-exposure camera, only exposure time was varied between frames; automatic exposure and automatic gain control were disabled. The sensor gain was fixed at 50, individual RGB gains were set to 0, and gain boost was disabled. Automatic white balance was disabled as well. Software gamma was set to 1.0, and hardware gamma was disabled.

Reference ECG signals were recorded at 500 Hz using a clinical-grade patient monitor (Mindray BeneVision N17) with a chest-lead configuration (Fig. 2(a)). The multi-exposure video sequences and ECG signals were timestamped to enable modal temporal alignment.

3) Acquisition Configurations: All evaluated methods provided an effective video output rate of 15 fps for rPPG signal extraction, although their acquisition rates and corresponding nominal exposure limits differed. Full-frame AE operated at 15 fps, allowing exposures up to approximately 66.6 ms. Triplet fitting acquired three successive frames at 45 fps for each output frame, with a nominal exposure limit of approximately 22.2 ms. The six-stage cyclic sequence was acquired at 90 fps, imposing a nominal per-frame exposure limit of approximately 11.1 ms. This limit applied to the fixedexposure baseline, Mertens fusion, and PixExpo, all of which used recordings from this sequence.

Scenario I: Baseline Comparison. Camera 1 used the manufacturer’s default full-frame auto-exposure mode, while Camera 2 acquired the six-stage cyclic exposure sequence. The fixed-exposure baseline used one pre-specified exposure substream, yielding a 15-fps video. Mertens fusion [36] and PixExpo each combined all six exposure levels to produce 15-fps outputs. The input frames for Mertens fusion were spatially registered to reduce inter-frame misalignment, whereas PixExpo operated on the original, unregistered sequence in the main comparison. This scenario compared PixExpo with full-frame AE, fixed exposure, and a representative perceptionoriented exposure-fusion method.

Scenario II: Comparison with Global Adaptive Exposure. Camera 1 implemented the triplet-frame global adaptive exposure method proposed in [35], while Camera 2 simultaneously acquired the six-stage cyclic sequence. This configuration enabled a comparison between global adaptive exposure and PixExpo under the same driving conditions.

Additional ROI-Based AE Experiment. A separate experiment with seven additional participants compared PixExpo with Face-ROI AE and Skin-ROI AE using two cameras operating concurrently. One camera acquired the six-stage PixExpo sequence, while the second camera alternated between the two ROI-based controllers, yielding a 15-fps stream for each controller. The known one-frame exposure actuation latency was explicitly accounted for in the acquisition schedule, allowing the two AE controllers to maintain independent exposure states without cross-interference. Face-ROI AE used the YuNet face bounding box, whereas Skin-ROI AE used bilateral cheek regions localized from the five YuNet facial landmarks, thereby reducing interference from hair around the forehead. For both methods, exposure was adjusted online to maintain the mean intensity of the corresponding ROI within a predefined target range. The target range and controller parameters were fixed for all participants and were determined solely from image brightness, without using ECG, rPPG signals, or HR estimation errors for tuning. The nominal 33-ms exposure limit imposed by the 30-fps acquisition was not binding, as logged exposures for both controllers remained at or below 23 ms.

![](images/f3f0c18d6adfcc398016b2192b34ac7a325f55775a61d1ac3b99714d0d55077d.jpg)

![](images/70b6a252f7b59a8a8b793c6ee2fdf970f80ef63a62a6c1509fb8e83da15bef2d.jpg)

![](images/3030c04dd9c1762b36fc9c32902c681648badfc7daa22846407b430178691f16.jpg)  
Fig. 3: Participant-level distributions of MAE, SNR, and SR for four exposure and fusion strategies $( n = 4 8 )$ . The internal box plots show the median (red line), mean (white dot), and interquartile range.

## B. Evaluation Metrics

Because PixExpo targets physiological signal preservation rather than perceptual HDR quality, evaluation is based on ECG-referenced HR accuracy and rPPG signal quality. Conventional HDR image-quality metrics are not reported because they primarily assess spatial or perceptual fidelity and do not directly measure preservation of pulsation-induced temporal color variations. We use the metrics defined in [35]:

• Mean absolute error (MAE): the mean absolute difference between the estimated HR and the ECG reference, measured in bpm.

• Success rate (SR): the percentage of HR estimates satisfying $| H R _ { \mathrm { e s t } } - H R _ { \mathrm { r e f } } | \leq 5$ bpm.

• Signal-to-noise ratio (SNR): the ratio of spectral power within the predefined neighborhoods of the reference HR and its first harmonic to the remaining power in the [0.7, 4.0] Hz evaluation band.

POS [37] served as the default rPPG estimator. To assess the performance consistency of PixExpo across different rPPG algorithms, EfficientPhys [38], FactorizePhys [39], Phys-Former [40], and iBVPNet [41] were additionally evaluated on all 48 MEX-Drive participants using PURE-pretrained checkpoints distributed by rPPG-Toolbox [42]. Each checkpoint was applied to PixExpo, Fixed Exposure, and Full-frame AE without training or fine-tuning on MEX-Drive. The separate seven-participant ROI-based AE experiment was evaluated using POS, CHROM [43], and ICA [44].

For the pretrained models, aligned RGB face crops were resized to the required spatial resolution and temporally resampled from 15 to 30 fps by linear interpolation, preserving video duration and matching the checkpoint configurations. Modelspecific normalization and temporal chunking were then applied. For each estimator, facial ROI processing, preprocessing, temporal windows, signal filtering, and HR estimation settings were kept identical across the compared acquisition strategies.

## V. RESULTS AND DISCUSSION

## A. Scenario I: Comparison with Baseline Strategies

1) Comparison with Conventional Exposure Strategies: We first compared PixExpo with full-frame AE, the fixedexposure substream, and registered Mertens fusion across all

48 participants. The Mertens inputs were spatially registered before fusion to reduce the effect of inter-frame misalignment. As shown in Fig. 3, PixExpo yielded the lowest mean MAE and the highest mean SNR and SR among the four strategies.

Specifically, PixExpo achieved an MAE of $6 . 7 3 { \pm } 3 . 5 6 \mathrm { b p m }$ an SNR of $0 . 6 0 { \pm } 3 . 2 7 { \mathrm { d B } }$ , and an SR of $6 3 . 2 4 \pm 1 8 . 9 5 \%$ . Fullframe AE yielded $1 3 . 9 4 \pm 6 . 4 4 { \mathrm { b p m } } , \ - 4 . 3 9 \pm 2 . 5 5 { \mathrm { d B } }$ , and $2 5 . 9 5 \pm 1 8 . 3 1 \%$ , respectively. The fixed-exposure substream yielded $9 . 9 2 \pm 5 . 0 4 \mathrm { b p m } , - 1 . 4 7 \pm 3 . 2 1 \mathrm { d B }$ , and $4 7 . 8 0 { \pm } 2 1 . 0 0 \%$ while registered Mertens fusion yielded $9 . 6 2 \pm 4 . 2 7 \mathrm { b p m } .$ $- 1 . 5 7 \pm 3 . 5 1 \mathrm { d B }$ , and $5 0 . 1 5 \pm 2 2 . 8 9 \%$ . Relative to Mertens fusion, PixExpo reduced MAE by 30.1%, increased SNR by 2.17 dB, and increased SR by 13.09 percentage points. The participant-level distributions in Fig. 3 are correspondingly shifted toward lower MAE and higher SNR and SR.

Fig. 4 provides an example. Full-frame AE produces extensive facial saturation, due to the dark cabin background driving the global controller toward a longer exposure. The fixedexposure frame avoids severe global saturation but retains a dark forehead and bright cheeks. Registered Mertens fusion produces a more visually balanced face, whereas PixExpo exhibits visible exposure-selection boundaries. Despite its less uniform appearance, PixExpo provides the clearest timefrequency component around the reference HR in this example. It achieves an MAE of 3.52 bpm, an SNR of 1.97 dB, and an SR of 76.98%, compared with an MAE of 7.53 bpm, an SNR of −1.64 dB, and an SR of 53.97% for Mertens fusion. Because the Mertens inputs were registered before fusion, this difference cannot be attributed solely to inter-frame misalignment. Instead, the result suggests that perceptual smoothness and preservation of pulse-related temporal variations are not equivalent objectives.

Taken together, the participant-level distributions and the example show that PixExpo provides more favorable rPPG measurements under spatially and temporally varying invehicle illumination, although its fused frames may be less visually smooth than conventional MEF outputs.

Cross-Dataset Evaluation with Pretrained rPPG Models. Table II presents results for 4 PURE-pretrained rPPG models evaluated on 48 MEX-Drive participants without fine-tuning. PixExpo achieved the lowest MAE and RMSE and the highest SR for every evaluated model. Relative to Fixed Exposure and Full-frame AE, its MAE reductions ranged from 23.1% to

(i)  
(a) Full-frame AE  
![](images/1cd2e0fc35ef8dff59ccf70260b26e8376b5acb32b3224ef9d6031cc2c2aa067.jpg)

(b) Fixed exposure  
![](images/0c667c0f59a1c3dfa5359a2c7cea079e92ce3e176821c14e746c6be34cb4da53.jpg)

(c) Mertens fusion  
![](images/0c6e0ed85729cef513d26eb3bf13d8a49f27c78165e91e7fbd4c6c5a009a8e72.jpg)

![](images/7bffdf6b913358ea99a979c59a595e2e537f47c5eab9ed592b789681cab7b0ec.jpg)  
Fig. 4: Visual and time-frequency comparison of four exposure and fusion strategies. The first row shows representative frames, and the second row shows the corresponding time-frequency representations. The black dashed curve denotes the estimated HR, and the white curve denotes the ECG-derived reference HR. HR MAE (bpm), SNR (dB), and SR (%) are reported below each method.

42.0% and from 22.8% to 53.9%, respectively. These results show that the acquisition-related benefit of PixExpo extends to the evaluated learned estimators under cross-dataset testing.

TABLE II: Cross-dataset HR estimation performance on all 48 MEX-Drive participants using PURE-pretrained rPPG models without fine-tuning. Bold values indicate the best result among the three acquisition strategies for each estimator and metric.
<table><tr><td></td><td colspan="3">PixExpo</td><td colspan="3">Fixed Exposure</td><td colspan="3">Full-frame AE</td></tr><tr><td>Estimator</td><td>MAE↓ RMSE↓</td><td></td><td>SR↑</td><td>MAE↓</td><td>RMSE↓</td><td>SR↑</td><td>MAE↓ RMSE↓</td><td></td><td>SR↑</td></tr><tr><td>EfficientPhys [38]</td><td>8.30</td><td>13.30</td><td>57.91</td><td>10.80</td><td>15.38</td><td>43.87</td><td>12.05</td><td>16.72</td><td>38.21</td></tr><tr><td>FactorizePhys [39]</td><td>4.11</td><td>7.76</td><td>79.44</td><td>7.09</td><td>11.36</td><td>61.31</td><td>8.91</td><td>13.36</td><td>51.04</td></tr><tr><td>PhysFormer [40]</td><td>8.12</td><td>12.93</td><td>58.14</td><td>10.56</td><td>15.82</td><td>47.10</td><td>10.52</td><td>15.24</td><td>45.27</td></tr><tr><td>iBVPNet [41]</td><td>8.38</td><td>13.51</td><td>61.76</td><td>11.08</td><td>15.99</td><td>46.12</td><td>12.10</td><td>16.71</td><td>40.61</td></tr></table>

2) Additional Comparison with ROI-Based AE: Fig. 5 reports the additional seven-participant comparison among the fixed-exposure baseline, Face-ROI AE, Skin-ROI AE, and Pix-Expo using three representative training-free rPPG algorithms: POS, CHROM, and ICA. The ROI-based AE methods generally improved upon fixed exposure, while PixExpo achieved the lowest mean MAE and the highest overall SNR and SR for all three algorithms.

The mean MAEs obtained by PixExpo were 3.63, 3.19, and 5.96 bpm for POS, CHROM, and ICA, respectively. The corresponding best ROI-based AE results were 4.21, 3.72, and 7.46 bpm. Similar relative trends are observed for SNR and SR in Fig. 5. Within this additional experiment, these results support that the improvement provided by PixExpo is general to rPPG algorithms, not specific to POS.

The remaining difference between PixExpo and ROI-based AE may be explained by the global exposure characteristics of the latter. A single exposure determined from an aggregate ROI statistic cannot independently accommodate bright and dark facial regions. Closed-loop AE may also respond to transient changes in ROI localization or illumination, causing short-term exposure fluctuations. PixExpo instead selects among multiple exposure observations at each pixel location and is therefore less dependent on a single global exposure setting.

![](images/11e854237e73e0b79adf181a3c7fcd2664aa9ec3251b54fbece262fc92a6cf7a.jpg)

![](images/3bb97ae63172af3bc847b118710a649769d72c1490d1803c75a6696cd6afbee3.jpg)  
(e)

![](images/d861f576efc3584fafb150d8235c92a35801922781ed954e4850f87bfa6ec5fb.jpg)

![](images/b58f16b9e80b0e783ae5900759a0d59ae1c83628abe38181fe63b66dc5a7a664.jpg)

(f)  
![](images/6ee1c8c0cf7e6793aff02b482f23c3ae8e5901b1bc4fa213eed2a5287385ce2f.jpg)  
(h)

![](images/d9754dad3bda0a886cbd3fa9240e42b1e789fcb7d165083749ecc7932d5b3388.jpg)

![](images/4db23c0e9d987baa4759bd331f6a4916611b196e48064b106e605d3c2cec8e6e.jpg)

![](images/9e628ffd8101c59856bebc7bb0d04237885fa498cc8f687c5aca0ce00100f469.jpg)

![](images/51bf2793630f005b243c7bcdde6617119b24acdc1d8951bdc406a080db7fbbe4.jpg)  
Fig. 5: Performance comparison of Fixed Exposure, Face-ROI AE, Skin-ROI AE, and PixExpo using POS, CHROM, and ICA in the additional seven-participant experiment. Light lines connect paired participant-level results, and black markers show the mean ± SEM.

B. Scenario II: Comparison with Fixed and Adaptive Global Exposure

1) Overall Performance: We compared PixExpo with fixed exposure and the global adaptive exposure method in [35] across all 48 participants. As shown in Fig. 6, PixExpo yielded the lowest mean MAE and the highest mean SR among the three strategies.

PixExpo achieved an MAE of 6.30 ± 3.39 bpm, compared with $7 . 7 9 \pm 3 . 6 9$ bpm for global adaptive exposure and $9 . 7 5 \pm 4 . 6 1$ bpm for fixed exposure. These values correspond to reductions in mean MAE of 19.1% and 35.4%, respectively. PixExpo also achieved an SR of $6 5 . 2 4 \pm 1 8 . 0 4 \%$ , compared with $5 9 . 7 0 { \pm } 1 6 . 5 3 \%$ for adaptive exposure and $4 9 . 9 4 { \pm } 2 0 . 7 8 \%$ for fixed exposure, representing increases of 5.54 and 15.30 percentage points, respectively.

The paired participant-level trajectories in Fig. 6 show that

![](images/9ab99b17cc86e545efb88580cda5450285e8c3b0006ba85915534bddee6f6ccc.jpg)

![](images/f3981013107eb605abf413ff3232b836af19c2c0988d58659305d0f0a711c9db.jpg)  
Fig. 6: Participant-level comparison of fixed exposure, global adaptive exposure, and PixExpo in Scenario II $( n = 4 8 )$ . Grey lines connect paired results from the same participant; violin plots show the corresponding distributions of MAE and SR.

PixExpo reduces MAE and increases SR for most participants relative to global adaptive exposure. However, the improvement is not uniform: several participants exhibit a higher MAE or a lower SR with PixExpo. These cases indicate that the performance of the fixed six-exposure configuration depends on the acquisition conditions. We therefore further examine the results under different illumination conditions.

2) Performance under Different Illumination Conditions: To examine how illumination affects the relative performance of the three strategies, we divided the recordings into rainy/overcast (n = 12) and sunny (n = 36) conditions. The subgroup means are shown in Fig. 7.

![](images/627369b122b7d02ba8639e191813bb84c91dc45a6cc78e290ee7482b9869d3ae.jpg)

![](images/bad3793b151ee51086d8ce64ac0bcb0d5962f04ce17703ff07c10b4cb2f569b1.jpg)  
Fig. 7: Mean MAE and SR of fixed exposure, global adaptive exposure, and PixExpo under rainy and sunny conditions.

Under sunny conditions, PixExpo achieved a mean MAE of 6.20 bpm and an SR of 67.04%, compared with 8.19 bpm and 59.42% for global adaptive exposure and 10.28 bpm and 48.36% for fixed exposure. Direct sunlight frequently produced strong spatial illumination gradients across the face. In these conditions, the six-exposure sequence allowed PixExpo to select shorter-exposure observations for strongly illuminated regions and longer-exposure observations for darker regions, reducing localized saturation and underexposure.

Under rainy or overcast conditions, PixExpo achieved a mean MAE of 6.80 bpm and an SR of 59.20%. Its performance was slightly lower than that of global adaptive exposure, which achieved an MAE of 6.61 bpm and an SR of 60.64%, although it remained better than fixed exposure in terms of both metrics.

A plausible explanation is the exposure-time constraint imposed by the fixed six-channel configuration. Maintaining 15 fps for each of the six exposure channels required a total acquisition rate of 90 fps, limiting the nominal maximum exposure time to approximately 11.1 ms. The triplet-frame adaptive method operated at 45 fps and therefore allowed a nominal maximum exposure time of 22.2 ms. This longer integration time is advantageous when photon availability is limited. These results suggest a tradeoff between spatial exposure coverage and per-frame photon accumulation, motivating an illumination-adaptive choice of exposure-channel count.

3) Representative Case Studies: Fig. 8 illustrates the behavior of fixed exposure, global adaptive exposure, and PixExpo in four representative in-vehicle conditions.

![](images/46bc1f6766f589d949c54d9e9dc78c7575beb038ac8cd65c127019531fb08f5c.jpg)  
Fig. 8: Representative video frames and corresponding rPPG spectrograms under four in-vehicle illumination conditions. Rows 1–4 show sunset, sunny with head rotation, overcast, and rainy conditions, respectively. Columns (a)–(c) show fixed exposure, global adaptive exposure, and PixExpo. The MAE and SR are reported below each spectrogram.

Case 1: Direct Sunset with Strong Vertical Illumination Gradients. Direct sunset produces a strongly illuminated lower face while the upper region remains shadowed by the vehicle roof. Fixed exposure results in extensive saturation and an unstable HR trace, with an MAE of 9.84 bpm and an SR of 52.9%. Global adaptive exposure moves the average facial intensity toward $I _ { \mathrm { t a r g e t } } = 1 4 0$ and improves the result to an MAE of 6.79 bpm and an SR of 70.6%, but localized saturation remains around the chin and jawline. PixExpo selects shorterexposure observations in the strongly illuminated regions, reducing local clipping and yielding an MAE of 2.44 bpm and an SR of 90.2%.

Case 2: Sunny Conditions with Head Rotation. In this case, head rotation exposes a larger cheek region to direct illumination. Fixed exposure exhibits extensive cheek saturation and achieves an MAE of 12.5 bpm and an SR of 27.1%. Global adaptive exposure reduces the overall exposure time, but residual local saturation remains, resulting in an MAE of 9.65 bpm and an SR of 35.7%. Pixel-wise one-hot selection across the six exposure levels reduces this saturation and improves the MAE to 3.56 bpm and the SR to 78.7%. Although the fused frame contains visible exposure-selection boundaries, these artifacts do not dominate the recovered HR trace in this example.

Case 3: Stable Overcast Conditions. Under relatively stable overcast illumination, all three strategies provide usable HR estimates. Fixed exposure exhibits transient deviations during changes in ambient illumination and achieves an MAE of 6.80 bpm and an SR of 71.7%. Global adaptive exposure produces a more continuous HR trace, with an MAE of 3.25 bpm and an SR of 81.6%. PixExpo yields the lowest MAE of 1.59 bpm and the highest SR of 93.6% in this example.

Case 4: Low-Light Rainy Conditions. This case illustrates a limitation of the fixed six-exposure configuration. The 90- fps acquisition rate limited the nominal maximum exposure time to 11.1 ms, reducing photon accumulation under low illumination. PixExpo achieved an MAE of 4.89 bpm and an SR of 74.3%, compared with 3.53 bpm and 77.7% for fixed exposure and 3.22 bpm and 78.7% for global adaptive exposure. Thus, the six-exposure configuration did not provide an advantage in this low-light example.

4) Effect of Exposure Channel Count under Low Light: To examine whether the low-light result was related to the exposure budget, we conducted an additional controlled experiment with seven participants using $N = 2 , 3$ , and 6 exposure channels. Each configuration was newly acquired rather than generated by subsampling the six-channel recordings. The illumination, camera settings, fusion criterion, and downstream rPPG processing were kept unchanged.

As shown in Table III, N = 2 achieved the lowest mean MAE of 3.75 bpm, the highest mean SNR of 4.92 dB, and the highest mean SR of 90.15%. In comparison, $N = 6$ yielded an MAE of 6.99 bpm, an SNR of 1.19 dB, and an SR of 67.76%. Reducing the channel count from six to two decreased MAE by 46.4%, increased SNR by 3.73 dB, and increased SR by 22.39 percentage points.

These results support the explanation that a fixed sixchannel configuration can become suboptimal under low illumination because the higher total frame rate restricts the available integration time. They motivate future selection of the exposure-channel count according to scene illumination.

TABLE III: Effect of Exposure Channel Count under Controlled Low-Light Conditions
<table><tr><td>N</td><td>Total rate (fps)</td><td>Exposure times (ms)</td><td>MAE (bpm)</td><td>SNR (dB)</td><td>SR (%)</td></tr><tr><td>2</td><td>30</td><td> $\overline { { 1 6 , 3 2 } }$ </td><td>3.75</td><td>4.92</td><td>90.15</td></tr><tr><td>3</td><td>45</td><td>7, 14, 21</td><td>4.59</td><td>3.30</td><td>81.76</td></tr><tr><td>6</td><td>90</td><td> $1 . 8 \times \{ 1 , \dots , 6 \}$ </td><td>6.99</td><td>1.19</td><td>67.76</td></tr></table>

However, the seven-participant experiment does not establish that $N = 2$ is universally optimal for all low-light conditions.

## C. Ablation Study on Fusion Weighting Strategies

To evaluate the fusion weighting strategy, we compared onehot, Gaussian-decay, and inverse weighting in all 48 participants while keeping the multi-exposure inputs and downstream rPPG pipeline unchanged. For Gaussian-decay and inverse weighting, $\sigma ~ = ~ 1$ and $\epsilon \ = \ 1$ , respectively, were fixed across participants without subject-specific tuning. As shown in Table IV, one-hot weighting yielded the lowest MAE (6.73 ± 3.56 bpm) and the highest SR $( 6 3 . 2 4 \pm 1 8 . 9 5 \% )$ and SNR $( 0 . 6 0 \pm 3 . 2 7 \mathrm { d B } )$ . Relative to Gaussian-decay weighting, it reduced MAE by 11.0%, increased SR by 5.34 percentage points, and increased SNR by 0.45 dB. Friedman tests showed differences among the three strategies for all metrics (all $p \ < \ 0 . 0 0 1 )$ , with Kendall’s W values of 0.365, 0.730, and 0.875 for MAE, SR, and SNR, respectively. Holm-corrected post-hoc Wilcoxon signed-rank tests further showed that onehot weighting outperformed each soft-weighting strategy for all three metrics (all adjusted $p < 0 . 0 0 1 )$ .

TABLE IV: Participant-level Comparison of Fusion Weighting Strategies (n = 48)
<table><tr><td>Fusion Strategy</td><td>MAE (bpm)↓</td><td>SR (%)↑</td><td>SNR (dB)↑</td></tr><tr><td>One-hot</td><td> ${ \bf 6 . 7 3 \pm 3 . 5 6 }$ </td><td> ${ \bf 6 3 . 2 4 \pm 1 8 . 9 5 }$ </td><td> ${ \bf 0 . 6 0 } \pm 3 . 2 7$ </td></tr><tr><td>Gaussian decay</td><td> $7 . 5 6 \pm 3 . 7 7$ </td><td> $5 7 . 9 0 \pm 1 9 . 2 5$ </td><td> $0 . 1 5 \pm 3 . 2 3$ </td></tr><tr><td>Inverse weighting</td><td> $9 . 8 3 \pm 4 . 5 1$ </td><td> $4 7 . 4 3 \pm 2 0 . 7 3$ </td><td> $- 1 . 3 4 \pm 3 . 5 2$ </td></tr><tr><td>Friedman p</td><td> $< 0 . 0 0 1$ </td><td>&lt; 0.001</td><td>&lt; 0.001</td></tr><tr><td>Kendall&#x27;s W</td><td>0.365</td><td>0.730</td><td>0.875</td></tr></table>

![](images/225198b7df4595b90b6c273df1efdb7a980b113da36fb48024635d55d3293957.jpg)

![](images/ada0e356d999bfcec228809aebff07223ccd6f03e604d4562cfc5efb959cd185.jpg)

![](images/1021945aac856e12406e8a46c0acff548d8a6b05dc6013c30e301da3503cc878.jpg)  
Fig. 9: Visual and time–frequency comparison of fusion weighting strategies.

![](images/ac3dd8017680d43e160c2d1fcbc686eb49fcf6c78e9dcc7030c320e15b8c1565.jpg)  
Fig. 10: Representative comparison under transient driving disturbances. The six frames shown above are representative frames selected from the time interval highlighted by the red dashed rectangle in each row.

Fig. 9 presents a representative comparison. One-hot weighting produces sharper local intensity transitions, whereas Gaussian-decay weighting yields a smoother appearance and inverse weighting retains stronger illumination contrast. However, the time–frequency representations and physiological metrics show that greater visual smoothness does not necessarily improve rPPG performance. In the POS-based evaluation, RGB values are spatially averaged over the facial ROI before pulse extraction; smooth transitions between neighboring pixels are therefore not a prerequisite for obtaining a reliable aggregate signal. Soft weighting combines multiple temporally offset candidates, including observations ranked farther from the target intensity, which may dilute the pulsatile component. In contrast, one-hot weighting retains only the highest-ranked candidate at each pixel. These results support the use of one-hot weighting when rPPG fidelity, rather than perceptual smoothness, is the primary objective.

## D. Robustness Analysis and Limitations

1) Robustness to Motion and Illumination Changes: In practical driving scenarios, head motion, vehicle vibration, and rapid illumination changes may occur within one exposure cycle. Since the six exposure levels are acquired at 90 fps, one complete cycle lasts only about 66.7 ms, which limits the spatial displacement between consecutive frames under typical driving conditions. Moreover, rPPG estimation is usually based on spatially aggregated RGB signals over the skin ROI, so small local misalignments are further attenuated by spatial averaging. Fig. 10 shows a representative segment containing head motion, vehicle vibration, and rapid illumination variation. Compared with fixed exposure and camera auto-exposure, PixExpo maintains a dominant spectral component close to the reference HR, demonstrating that it remains robust under these transient disturbances.

2) Effect of Inter-frame Registration: Sequential multiexposure acquisition introduces temporal offsets and potential spatial inconsistencies among the exposure candidates. Conventional multi-exposure fusion jointly blends multiple frames;

therefore, misregistration can produce duplicated contours or superposition-type ghosting, as illustrated in Fig. 11(a). PixExpo instead selects only one exposure candidate for each output pixel. Consequently, misregistration is more likely to produce local exposure-selection discontinuities than duplicated contours.

![](images/f374c9fe4ad5d998f98f7b26690503ff8b961dd9b846fe3673250561c0f18c7f.jpg)  
(a) Mertens fusion without registration

![](images/e5bdd8f8a968056105dd317a898d01e067e78fbb9ae781ea1b918c93e2aee6c8.jpg)  
(b) Mertens fusion with registration

![](images/37d83f0fbb79b768c39d797637f32d5ba7b5090472c4d7cc4cafb0589380c41d.jpg)  
(c) PixExpo fusion without registration

![](images/6dd7ec86db7ce7231635d95a1bc5a806057d24fde5ea4116999bb4c3915285ba.jpg)  
(d) PixExpo fusion with registration  
Fig. 11: Visual effect of inter-frame registration on Mertens fusion and PixExpo in a representative example.

As shown in Fig. 11, registration visibly reduced ghosting in Mertens fusion, whereas the PixExpo outputs before and after registration remained similar. In the 14-participant subset, the mean HR MAE was $7 . 2 6 \pm 3 . 8 3$ bpm without registration and 7.41 ± 3.83 bpm with registration. No significant differences were found in MAE, SR, or SNR (all $p > 0 . 0 5 )$ . These results suggest that registration provided no measurable rPPG benefit under the tested conditions, although more abrupt motion warrants further evaluation.

3) Temporal Characteristics of Exposure Selection: Under one-hot selection, the selected exposure channel may switch over time as changes in illumination or subject movement alter the candidate intensities. Switching is particularly likely near facial boundaries and high-gradient regions, where even small spatial displacements can change pixel intensities enough to alter the selected exposure label. Consistent with this explanation, the pixel switching rate (PSR) map in Fig. 12(a) shows more frequent switching in these regions.

Frequent switching, however, does not necessarily produce periodic artifacts within the pulse-frequency range. Fig. 12(b) compares the power spectral densities (PSDs) of the frame switching ratio (FSR), the mean selection index, and the recovered rPPG signal. FSR measures the fraction of pixels whose exposure labels change between consecutive fused frames, while the mean selection index summarizes the overall exposure-selection state. Neither switching descriptor exhibits a pronounced narrowband peak comparable to that of the recovered rPPG signal, whose peak at 1.52 Hz is close to the mean reference HR expressed in frequency units (1.56 Hz). These observations suggest that exposure switching is not a dominant source of pulse-like artifacts in this recording.

![](images/c8de2020fcce043add851dfd8dad4176084ae7e7dcf3dfffb58d8ac798294e11.jpg)

![](images/7d016e6e8d90a5073ef15008e50876bf42380ab8440eae9b142d999808fedbd2.jpg)  
Fig. 12: Spatial and spectral characteristics of exposure switching in a representative example. (a) PSR map, showing higher switching rates around facial boundaries and other highgradient regions and lower switching rates over the forehead and cheeks. (b) PSDs of the recovered rPPG signal, FSR, and mean selection index. The recovered rPPG signal peaks at 1.52 Hz (91.0 bpm), close to the mean reference HR of 1.56 Hz.

4) Limitations and Future Work: The current implementation uses a fixed stack of six exposure levels. Although this configuration covers a broad dynamic range, the controlled low-light experiment in Sec. V-B4 showed that it can be suboptimal under low illumination. Maintaining a 15-fps output with six exposure channels requires an aggregate acquisition rate of 90 fps, thereby limiting the maximum integration time to approximately 11.1 ms. This constraint can reduce photon accumulation and temporal SNR in dark scenes. Future work will therefore investigate illumination-aware adjustment of both the number and distribution of exposure levels.

Although PixExpo targets dynamic illumination, its performance across skin tones remains to be established. The current cohort covers only MST levels 4–6, limiting the generalizability of the findings to other skin-tone groups. Future work should include participants spanning a broader range of skin tones and evaluate performance through skintone-stratified analyses.

## VI. CONCLUSIONS

This study introduced PixExpo, a computational framework that combines cyclic multi-exposure acquisition with rPPGoriented pixel-wise fusion to mitigate spatially nonuniform illumination in vehicle cabins. Without modifying the image sensor, PixExpo selects at each pixel the exposure observation closest to a target intensity. On MEX-Drive, which included 48 participants, PixExpo reduced HR MAE from 13.94 to 6.73 bpm and increased SR from 25.95% to 63.24% relative to manufacturer-default auto-exposure. The additional experiments also showed better mean performance than global adaptive exposure under the tested conditions and favored onehot selection over Gaussian-decay and inverse weighting.

The results also reveal an acquisition tradeoff. Maintaining a 15-fps output with six exposure channels requires a 90- fps acquisition rate and limits the maximum integration time to approximately 11.1 ms, which can impair performance under low illumination. Future work will adapt the number and distribution of exposure levels to scene illumination and validate PixExpo across broader skin-tone groups, additional cameras, and more severe motion conditions. Overall, PixExpo provides a practical acquisition and fusion strategy for improving in-vehicle rPPG under spatially nonuniform illumination, provided that frame-level exposure control and a sufficiently high acquisition rate are available.

## REFERENCES

## REFERENCES

[1] B. Huang, S. Hu, Z. Liu, C.-L. Lin, J. Su, C. Zhao, L. Wang, W. Wang, Challenges and prospects of visual contactless physiological monitoring in clinical study, npj Digit. Med. 6 (2023).

[2] J. Wang, C. Shan, L. Liu, Z. Hou, Camera-based physiological measurement: Recent advances and future prospects, Neurocomputing 575 (2024) 127282.

[3] S. G. Ahmed, K. Verbert, N. Zaki, A. Khalil, H. Aljassmi, F. Alnajjar, Ai innovations in rppg systems for driver monitoring: Comprehensive systematic review and future prospects, IEEE Access 13 (2025) 22893– 22918.

[4] A. Gupta, A. G. Ravelo-Garc´ıa, F. M. Dias, RGB, a surrogate of infrared facial videos for physiological signs estimations in dark, IEEE Trans. Circuits Syst. Video Technol. 36 (5) (2026) 6499–6512.

[5] J. Laurie, N. Higgins, T. Peynot, J. Roberts, Dedicated exposure control for remote photoplethysmography, IEEE Access 8 (2020) 116642– 116652.

[6] J. Kim, Y. Cho, A. Kim, Proactive camera attribute control using bayesian optimization for illumination-resilient visual navigation, IEEE Trans. Robot. 36 (4) (2020) 1256–1271.

[7] S. Yi, D. Yu, Y. Zhu, W. Wang, Illumination-robust camera-ppg via skin-guided auto-exposure, in: EMBC, 2024, pp. 1–4.

[8] X. Zhang, Benchmarking and comparing multi-exposure image fusion algorithms, Inf. Fusion 74 (2021) 111–131.

[9] F. Xu, J. Liu, Y. Song, H. Sun, X. Wang, Multi-exposure image fusion techniques: A comprehensive review, Remote Sens. 14 (3) (2022) 771.

[10] Y. Li, M. Liu, K. Han, Overview of multi-exposure image fusion, in: ICEIB, 2021, pp. 196–198.

[11] Z. Wang, X. Yang, H. Lu, C. Shan, W. Wang, Benchmark of physiological model based and deep learning based remote photoplethysmography in automotive applications, in: ICASSP, 2023, pp. 1–5.

[12] T. Arakawa, A review of heartbeat detection systems for automotive applications, Sensors 21 (18) (2021).

[13] S. Leonhardt, L. Leicht, D. Teichmann, Unobtrusive vital sign monitoring in automotive environments—a review, Sensors 18 (9) (2018).

[14] M. H. Cho, S. G. Lee, B. D. Nam, Fast auto exposure algorithm based on the numerical analysis, in: SPIE Sens., Cameras, Appl. Digit. Photogr., Vol. 3650, 1999, pp. 93–99.

[15] Y. Su, J. Y. Lin, C.-C. J. Kuo, A model-based approach to camera’s auto exposure control, J. Vis. Commun. Image Represent. 36 (2016) 122–129.

[16] Q. K. Vuong, S.-H. Yun, S. Kim, A new auto exposure system to detect high dynamic range conditions using CMOS technology, in: ICCIT, Vol. 1, 2008, pp. 577–580.

[17] L. Hongqin, W. Jianzhen, Z. Liping, N. Jun, Real-time digital image exposure status detection and circuit implementation, Int. J. Adv. Comput. Sci. Appl. 6 (5) (2015) 19–24.

[18] I. Shim, T.-H. Oh, J.-Y. Lee, J. Choi, D.-G. Choi, I. S. Kweon, Gradientbased camera exposure control for outdoor mobile platforms, IEEE Trans. Circuits Syst. Video Technol. 29 (6) (2019) 1569–1583.

[19] I. Mehta, M. Tang, T. D. Barfoot, Gradient-based auto-exposure control applied to a self-driving car, in: CRV, 2020, pp. 166–173.

[20] B. Han, Y. Lin, Y. Dong, H. Wang, T. Zhang, C. Liang, Camera attributes control for visual odometry with motion blur awareness, IEEE/ASME Trans. Mechatronics 28 (4) (2023) 2225–2236.

[21] K. Xu, Q. Wang, H. Xiao, K. Liu, Multi-exposure image fusion algorithm based on improved weight function, Front. Neurorobot. 16 (2022) 846580.

[22] S. Li, X. Kang, Fast multi-exposure image fusion with median filter and recursive filter, IEEE Trans. Consum. Electron. 58 (2) (2012) 626–632.

[23] S. Li, X. Kang, J. Hu, Image fusion with guided filtering, IEEE Trans. Image Process. 22 (7) (2013) 2864–2875.

[24] R. Shen, I. Cheng, J. Shi, A. Basu, Generalized random walks for fusion of multi-exposure images, IEEE Trans. Image Process. 20 (12) (2011) 3634–3646.

[25] S. Raman, S. Chaudhuri, Bilateral filter based compositing for variable exposure photography, in: Eurographics, 2009.

[26] T. Mertens, J. Kautz, F. Van Reeth, Exposure fusion: A simple and practical alternative to high dynamic range photography, Comput. Graph. Forum 28 (1) (2009) 161–171.

[27] B. Gu, W. Li, J. Wong, M. Zhu, M. Wang, Gradient field multiexposure images fusion for high dynamic range image visualization, J. Vis. Commun. Image Represent. 23 (4) (2012) 604–610.

[28] X. Liu, Perceptual multi-exposure fusion, arXiv preprint arXiv:2210.09604Version 3, 5 Mar 2025 (2025).

[29] L. Fu, C. Zhou, Q. Guo, Q. Yan, R. Wang, B. Zeng, X. Guo, Autoexposure fusion for single-image shadow removal, in: CVPR, 2021, pp. 10571–10580.

[30] L. Qu, S. Liu, M. Wang, Z. Song, Rethinking multi-exposure image fusion with extreme and diverse exposure levels: A robust framework based on fourier transform and contrastive learning, Inf. Fusion 92 (2023) 389–403.

[31] W. Shen, M. Cheng, G. Lu, G. Zhai, L. Chen, M. S. Asif, Z. Gao, Spatial temporal video enhancement using alternating exposures, IEEE Trans. Circuits Syst. Video Technol. 32 (8) (2022) 4912–4926.

[32] I. Odinaev, J. W. Chin, K. Ho Luo, Z. Ke, R. H. SO, K. Long Wong, Optimizing camera exposure control settings for remote vital sign measurements in low-light environments, in: CVPRW, 2023, pp. 6086– 6093.

[33] Y. Gao, S. Yi, Y. Shen, W. Wang, Automatic dual-exposure for illumination-robust remote physiological monitoring, in: EMBC, 2025, pp. 1–4.

[34] C. Xiao, S. Yi, D. Yu, W. Wang, Dynamic exposure controlling and suppression for highly robust remote-ppg in fitness, in: EMBC, 2024, pp. 1–4.

[35] J. Wang, X. Cai, C. Shan, W. Wang, Optimize-at-capture: Highlyadaptive exposure controlling for in-vehicle non-contact heart-rate monitoring (2026). arXiv:2605.04397.

[36] T. Mertens, J. Kautz, F. Van Reeth, Exposure fusion, in: PG, 2007, pp. 382–390.

[37] W. Wang, A. C. den Brinker, S. Stuijk, G. de Haan, Algorithmic principles of remote PPG, IEEE Trans. Biomed. Eng. 64 (7) (2017) 1479–1491.

[38] X. Liu, B. Hill, Z. Jiang, S. Patel, D. McDuff, EfficientPhys: Enabling simple, fast and accurate camera-based cardiac measurement, in: WACV, 2023, pp. 5008–5017.

[39] J. Joshi, S. S. Agaian, Y. Cho, FactorizePhys: Matrix factorization for multidimensional attention in remote physiological sensing, in: NeurIPS, Vol. 37, 2024, pp. 96607–96639.

Jieying Wang received the B.E. degree in electronic and information engineering from China Jiliang University, China, in 2015, and the M.S. in marine information science and engineering from Zhejiang University, China, in 2018, and the Ph.D. degree in computer science and technology from University of Groningen, the Netherlands, in 2022. She is currently a lecture with Shandong University of Science and Technology, China. Her research interests are image processing and machine learning, with applications in health monitoring.

![](images/db105468fa59d8010b6f88f4302991f7ece29e54d691e0e83d8f05cab506d7a0.jpg)

[40] Z. Yu, Y. Shen, J. Shi, H. Zhao, P. H. S. Torr, G. Zhao, PhysFormer: Facial video-based physiological measurement with temporal difference transformer, in: CVPR, 2022, pp. 4186–4196.

[41] J. Joshi, Y. Cho, iBVP dataset: RGB-thermal rPPG dataset with high resolution signal quality labels, Electronics 13 (7) (2024) 1334.

[42] X. Liu, G. Narayanswamy, A. Paruchuri, X. Zhang, J. Tang, Y. Zhang, R. Sengupta, S. Patel, Y. Wang, D. McDuff, rPPG-Toolbox: Deep remote PPG toolbox, in: NeurIPS, Vol. 36, 2023.

[43] G. de Haan, V. Jeanne, Robust pulse rate from chrominance-based rppg, IEEE Trans. Biomed. Eng. 60 (10) (2013) 2878–2886.

[44] M.-Z. Poh, D. J. McDuff, R. W. Picard, Non-contact, automated cardiac pulse measurements using video imaging and blind source separation, Opt. Express 18 (10) (2010) 10762–10774.

![](images/9555ca1490b82bee8df5476008837a6a9eb4e373ae832579bb2fa73605967d7e.jpg)

Cai Xinqi is currently an undergraduate student majoring in Biomedical Engineering at Southern University of Science and Technology (SUSTech), Shenzhen, China. Her research interests include exposure algorithms and health monitoring.

![](images/84dc5e59f644386f3abb6b98923b3b66a8d3c48ea33109994e093e060c9f06df.jpg)

Caifeng Shan is a full professor with Nanjing University, China. He was previously a Senior Scientist with Philips Research, Eindhoven, The Netherlands. He received the B.Eng. degree from the University of Science and Technology of China, the M.Eng. degree from the Institute of Automation, Chinese Academy of Sciences, and the Ph.D. degree from Queen Mary, University of London. His research interests include computer vision, pattern recognition, medical image analysis, and related applications. He has co-authored about 200 papers and more than 100

patent applications. He has served as Associate Editor for journals including Pattern Recognition, IEEE Journal of Biomedical and Health Informatics, and IEEE Transactions on Circuits and Systems for Video Technology. He is a Senior Member of IEEE.

![](images/7047019b1c661515df678773d8a679a7119c2c9847ee7638858a123d8664189f.jpg)

Wenjin Wang is an Associate Professor of Southern University of Science and Technology, China. He received the B.Sc. degree from Northeastern University, China (2011), the M.Sc. degree from University of Amsterdam, The Netherlands (2013), and the Ph.D. degree from Eindhoven University of Technology (TU/e), The Netherlands (2017). He was an Assistant Professor of TU/e and a Scientist of Philips Research Eindhoven. He published 130 peerreviewed scientific papers, 3 academic books, and holds 45 granted patents. He got the Prize Paper

Award of IEEE-TBME 2022. He was awarded the National Excellent Young Scholars (Overseas) in 2022. His current research was supported by the National Key R&D Program of China and Natural Science Foundation of China. His research interests include the video health monitoring and its translation into a medical device.