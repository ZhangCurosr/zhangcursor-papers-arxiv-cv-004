# One Sensor, Whole Body — 3D Body Pose from a Single Consumer Earbud IMU

Zhilin Guo University of Cambridge United Kingdom

Hakan Aktas University of Cambridge United Kingdom

Boqiao Zhang<sup>∗</sup> University of Cambridge United Kingdom

Wenzhao Li University of Cambridge United Kingdom

Chenliang Zhou University of Cambridge United Kingdom

Oszkár Urbán<sup>∗</sup> University of Cambridge United Kingdom

Ali Senguel University of Cambridge United Kingdom

Siyu Hong University of Cambridge United Kingdom

Josef Bengtson<sup>∗</sup>   
Chalmers University of   
Technology   
Sweden

Kyle Fogarty University of Cambridge United Kingdom

Cengiz Oztireli<sup>✉</sup>   
University of Cambridge   
United Kingdom   
aco41@cam.ac.uk

## Abstract

Consumer earbuds already stream inertial motion data from the head, one of the most widely worn sensor locations on the body. We ask how much of the 3D body pose a single such head IMU can recover, and whether adding more consumer sensors actually helps. We build a multimodal capture pipeline that records fourview RGB-D video together with an AirPods head IMU and two Striv insole IMUs, synchronize the streams post-hoc, and generate pseudo-ground-truth with SAM 3D Body, yielding a 35-take singlesubject benchmark spanning gait, turning, vertical, everyday, and clinically inspired motions. Adapting two recurrent model families (IMUPoser and MobilePoser), we show that one head IMU recovers lower-body pose at 79.0 mm rigid-MPJPE and per-foot ground contact at 0.809 macro-F1, and that a causal variant retains most of this accuracy at streaming latency. In paired per-take significance tests across both families, adding the consumer foot IMUs never significantly improves pose and significantly degrades it in two of four model–split combinations; a mounting-bias probe and feet-only ab lation identify insole orientation quality, not foot placement, as the mechanism. Extending the output to a 20-joint full-body skeleton maps the boundary: gross distal-arm motion is partially recoverable from the head alone, proximal upper-body pose is not, and staged fine-tuning recovers the leg accuracy that naive joint training sacrifices to multi-task dilution. For learned pose from consumer wearables, sensor reliability, not sensor count, is the binding constraint here. For the devices tested, the earbud is its sweet spot. Code is available at https://github.com/ZhilinGuo/one-sensor-whole-body.

CCS Concepts

• Computing methodologies → Motion capture; Activity recognition and understanding.

## Keywords

wearable sensing, inertial motion capture, earbud IMU, sensor reliability, multimodal capture, pseudo-ground truth

## ACM Reference Format:

Zhilin Guo, Boqiao Zhang, Oszkár Urbán, Josef Bengtson, Hakan Aktas, Wenzhao Li, Siyu Hong, Kyle Fogarty, Chenliang Zhou, Ali Senguel, and Cengiz Oztireli. 2026. One Sensor, Whole Body — 3D Body Pose from a Single Consumer Earbud IMU. In The 6th International Workshop on Human-centric Multimedia Analysis (HUMA ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3841192. 3841753

## 1 Introduction

Earbuds are among the most pervasive inertial sensor platforms: hundreds of millions already wear a calibrated IMU on the head for hours a day. If that single stream suficed for useful 3D body pose, motion capture would need no suit, no straps, and no extra hardware. This paper asks two questions: how much ofthe body pose can one consumer head IMU recover, and, because extra sensors are cheap to add but costly to wear, does adding more consumer sensors actually help?

Most inertial pose systems assume far more instrumentation: six-IMU suits with a pelvis tracker [8, 11, 22, 23], VR headsets with 6-DoF SLAM positions and hand controllers [1, 2, 10], or phone– watch–earbud subsets evaluated on simulated data [15, 19]. Our setting is deliberately minimal and, to our knowledge, unoccupied on real data: a single, unified consumer earbud IMU: 3-DoF fused orientation and acceleration, no absolute position, no hand trackers. Answering these questions requires labels a mocap lab cannot provide for everyday earbud wear, so we frame it as a human-centric multimodal capture problem. We record 35 takes spanning seven motion scripts covering both entertainment and clinically relevant movements, each with four-view RGB-D video plus head and foot IMU streams. With no hardware sync and no mocap lab, we align the modalities post-hoc using signal-based temporal calibration and use SAM 3D Body [21] as pseudo-ground-truth (pseudo-GT), then evaluate IMUPoser-adapted [15] and MobilePoser-adapted [19] recurrent models on a nine-joint lower-body skeleton, an auxiliary foot-contact task, a streaming (causal) variant, and a 20-joint full body extension that maps the recoverability boundary. The study is a controlled pilot: one subject, one capture day, one earbud model, one insole product; all claims are within-participant and devicespecific.

The answers are sharper than expected. (i) The head alone is strong: 79.0 mm rigid-MPJPE on leave-one-run-out, 0.809 macro-F1 on per-foot contact, and a causal variant retaining most of this accuracy at one-frame latency. (ii) More sensors do not help: in paired per-take tests, adding two foot IMUs never significantly improves pose and significantly degrades it in two of four model–split combinations. (iii) The mechanism is reliability, not placement: a mounting-bias sweep and feet-only ablation trace the degradation to foot-orientation calibration, while the earbud’s clean raw gyro scope remains directly learnable. (iv) The recoverability boundary is sharp: gross distal-arm swing carries signal, proximal upper-body pose does not, and staged fine-tuning recovers the leg accuracy that naive joint widening sacrifices. We contribute the capture pipeline and benchmark, paired-significance evidence that one head IMU is the optimal configuration among the consumer devices tested, a reliability probe explaining why, and a delimitation of what a single head IMU can and cannot recover.

## 2 Related Work

Sparse-IMU full-body pose. Classic real-time methods (DIP, TransPose, PIP, and TIP) reconstruct full-body motion from six IMUs, one mounted at the pelvis or lower back [8, 11, 22, 23]. Recent methods relax the count: IMUPoser uses variable subsets from phones, watches, and earbuds [15]; MobilePoser targets 1–3 mobile IMUs [19]; others broaden layouts and generalization [17, 24]. Nearly all prior methods assume a pelvis sensor and raw gyroscopes, assumptions consumer earbuds and insoles cannot meet. Crucially, the commodity-device evaluations are synthetic (AMASS-simulated IMUs); we evaluate on real captured streams, where consumer noise and mounting variability, the phenomena at the heart ofour finding, actually occur.

Head-mounted minimal sensing. Dittadi et al. generate fullbody SMPL poses from a single head-mounted device [2]; HMD-Poser scales from HMD-only to HMD plus a few IMUs [1]; Avatar-Poser, AGRoL, EgoPoser, and DivaTrack hallucinate lower-body motion from head-and-hand trackers [3, 9, 10, 20]. These XR systems observe 6-DoF head position from SLAM, usually with hand controllers, far richer than a raw earbud IMU. Ear2Pos reconstructs full-body pose from two earbud IMUs with personalized bone lengths [18], and ProgIP uses head-plus-wrists [25]. We push minimalism to the limit: one unified head IMU stream (an earbud pair), no positions, no hands, no personalization, on real data.

Foot/insole and clinical gait. Feet are established cues for gait phase and contact [4–6, 16], but whether consumer insole IMUs stay reliable outside controlled straight-line locomotion is largely untested on real hardware.

![](images/f8581b7866ce66b75359c251fdc23852d14e8a577a290c7b30e961d571db72a2.jpg)  
Figure 1: One synchronized take from the four RealSense cameras; one view feeds SAM 3D Body pseudo-labeling, the others support qualitative checks and label fusion (Sec. 6).

Table 1: Motion protocol: five runs repeat seven scripts (35 takes).  
Seq Content   
S1 Standing, T-pose, normal/slow/fast straight walking, 180<sup>◦</sup> turns   
S2 In-place stepping, marching, single-leg stance, weight shifts   
S3 Side-steps, 360<sup>◦</sup> in-place turns, backwards walking, figure-8   
S4 Half/full squats, forward bends, lunges, sit-to-stand   
S5 Everyday movement: walk–sit–pick-up–reach   
<sup>S6</sup><sub>S7</sub> Clinical gait: turns, abrupt stop/restart, dual-task, sit-stand   
Upper-limb reaches, pronation/supination, strikes, arm-swing

Vision+IMU and pseudo-GT. Where DeepFuse adds inertial signals to stabilize multi-view pose [7], we use vision exclusively to create labels: SAM 3D Body supplies pseudo-GT meshes [21], AMASS supplies synthetic pretraining IMUs [14], and temporal alignment follows camera–IMU calibration from motion agreement [12]. No prior benchmark targets a single consumer earbud IMU with real multimodal capture and explicit sensor-reliability diagnostics.

## 3 Capture System and Dataset

The capture rig (Fig. 1) consists of four Intel RealSense D455e RGB-D cameras (848 × 480 at 30 FPS, color + depth), an AirPods earbud pair, exposed by iOS as a single fused head-motion stream, captured via a custom iOS/macOS app (fused attitude, device-frame acceleration, and gyroscope), and left/right Striv insole IMUs (fused Euler orientation plus acceleration, no raw gyroscope channel, ∼30 Hz over BLE). The head IMU is our primary sensor; the insoles test whether extra consumer sensors earn their keep. The protocol (Table 1) comprises five repeated runs of seven motion scripts: 35 takes spanning gait, stepping, turning, vertical motion, everyday composites, clinically inspired gait, and upper-limb coordination, enabling leave-one-run-out and leave-one-motion-out evaluation.

## Dataset statistics.

After processing and alignment, the benchmark contains 82,417 frames (45.8 minutes at 30 FPS) across the 35 takes, averaging 78.5 s each (38.6–137.1 s). SAM 3D Body yields a usable lower-body label for every frame, and each foot is in ground contact for 65% of frames (at least one foot for 73%). Each take stores the aligned IMU streams, nine lower-body joints plus the full 70-keypoint SAM output, per-foot contact labels, and a validity mask.

## 4 Alignment and Pseudo-Ground-Truth

We do not utilise hardware synchronization, only Unix-style timestamps with constant ofsets and slow drift. We place all streams on a common 30 Hz grid via timestamps, refine the head ofset by crosscorrelating AirPods gyroscope magnitude against SAM-derived head angular speed, and refine a shared foot ofset by matching Striv acceleration energy (heel-strike impacts) to SAM foot-speed minima. Across all 35 takes the recovered ofsets are tightly clustered $( - 2 . 6 9 \pm 0 . 0 6 { : }$ s head, $- 3 . 0 5 \pm 0 . 1 0 s \mathrm { f e e t } )$ , so misalignment is not silently absorbed into pose error. To generate pseudo groundtruth labels, we run SAM 3D Body on a selected camera view and convert its Momentum-Human-Rig output to a nine-joint lowerbody target (pelvis, hips, knees, ankles, feet); the full 70-keypoint output is retained for the full-body extension (Sec. 6). We treat these as pseudo-GT, acknowledging single-view mesh recovery is less accurate than marker-based mocap.

## 5 Benchmark Models and Protocol

As an initial baseline, we adapt the two-layer bidirectional LSTM from IMUPoser [15] to our nine-joint lower-body pose task. Each sensor contributes linear acceleration and a 3 × 3 orientation matrix (12-D per IMU; 36-D for head-plus-feet), with masked input blocks to support sensor configuration ablations. The model predicts 6D rotations for nine lower-body joints, whose positions are recovered via forward kinematics under SMPL-style conventions [13]. As a second baseline we adapt MobilePoser [19], keeping its two-stage design and objectives (joint-position RNN feeding a pose RNN teacher-forced noisy joints, smoothness/jerk penalties) retargeted to our sensors and lower-body output, and we attach an auxiliary foot-contact head to the IMUPoser-adapted backbone. We pretrain all models on synthetic IMU generated from CMU, BioMotionLab NTroje, and MPI HDM05 sequences in AMASS [14], then fine-tune on real pseudo-GT per fold. We evaluate under leave-one-run-out (held-out repeats of seen motion types) and leave-one-motion-out (held-out motion classes, the harder generalization test). Because per-frame Procrustes (PA-MPJPE) can favour a near-static mean pose, we emphasize rigid-MPJPE, min<sub>� � t</sub> $\begin{array} { r l } {  { \frac { 1 } { T J } \sum _ { t , j } \| s R \hat { p } _ { t , j } + \mathbf { t } - p _ { t , j } \| } } \end{array}$ <sub>2</sub>, which applies a single similarity transform $( s , R , \mathbf { t } )$ ) per sequence. We also report per-joint motion $\begin{array} { r } { { R } _ { j } ^ { 2 } = 1 - \sum _ { t } \Vert \hat { p } _ { t , j } - \hat { p } _ { t , j } \Vert ^ { 2 } / \sum _ { t } \Vert \hat { p } _ { t , j } - \bar { \hat { p } } _ { j } \Vert ^ { 2 } } \end{array}$ against the temporal mean ${ \bar { p } } _ { j } ,$ so a static prediction scores ≈ 0. Distal variants average knees, ankles, and feet, where foot sensors should matter most; PA-MPJPE and MPJVE are reported for completeness. Statistical protocol. Aggregate gaps between sensor configurations can hide fold noise, so every head-only vs. head+feet claim is backed by a paired per-take analysis: rigid-MPJPE values are paired across the 35 takes and tested with Wilcoxon signed-rank (paired �-test secondary), Holm-corrected across the four model×split com parisons, with efect sizes (median Δ, Cohen’s $d _ { z } )$ and win counts. Streaming and full-body variants. A causal IMUPoser-adapted model (unidirectional LSTM, same capacity) emits each frame us ing only past observations, the configuration a deployed earbud system would run. A full-body variant widens the output head to 20 SMPL joints, supervising the 17 with MHR70 pseudo-GT coun terparts and the three spine joints weakly with interpolated targets (never evaluated); single-view arm labels are the noisiest source, so we down-weight upper-body loss, anchor the per-window align ment on the legs, and re-solve group alignment per body group in evaluation.

Implementation. The IMUPoser-adapted model is a 512-unit twolayer BiLSTM (10.6M parameters); the MobilePoser-adapted model stacks two 256-unit two-layer BiLSTMs (5.3M total). Models train on 150-frame (5 s) windows with Adam: synthetic pretraining 40 epochs $( \ln { 3 } { \times } 1 0 ^ { - 4 } )$ , per-fold fine-tuning 20 epochs (lr $1 0 ^ { - 4 } )$ ). Because consumer calibration leaves a global frame/scale ambiguity, finetuning minimizes an L2 on forward-kinematics joints under a perwindow closed-form rotation-plus-scale alignment (retraining takes a few hours on one A100).

![](images/c9a0c5f592583618d288911cea597a1c2ce1ea9bd6ea596c3abaf53cc2c22a31.jpg)  
Figure 2: Paired per-take head-only (x) vs. head+feet (y) rigid-MPJPE across both model families and splits; points above the diagonal favour head-only (systematic under IMUPoseradapted, �<0.02 Holm; a tie-to-trend under MobilePoseradapted).

## 6 Results

One head IMU is enough. Table 2 summarizes the benchmark. Head-only is the best observed configuration in both families: 79.0 mm rigid-MPJPE and 0.49 distal $R ^ { \overset { \vartriangle } { } }$ on leave-one-run-out for IMUPoser-adapted, generalizing better than every multi-sensor configuration when motion classes are held out (92.6 vs. 107.4 mm; distal $R ^ { 2 } \ 0 . 2 7 \ \mathrm { v s } . - 0 . 0 1 )$ , and consistent across folds (79.0 ± 5.0 mm). A zero-input control (116.4 mm) and mean-pose prior (105.5 mm) confirm the model exploits real inertial signal. The result rests on both training stages (zero-shot transfer: 322.2 mm; no pretraining: 94.1 mm).

Streaming readiness. Replacing the bidirectional LSTM with a unidirectional one (same capacity, one-frame latency) yields 92.9 mm on leave-one-run-out and 91.5 mm on leave-one-motion-out: the streaming head-only model still beats the bidirectional head+feet configuration (95.8 and 107.4 mm). With ∼3,500 CPU frames/s, a real-time earbud pose stream is deployable today.

More sensors do not help: significantly. The paired per-take analysis (Fig. 2) is the paper’s central evidence. Across the 35 takes, head+feet never significantly beats head-only in any of the four model×split comparisons. Under IMUPoser-adapted, headonly wins 29/35 takes on leave-one-run-out (median Δ +11.5 mm, $d _ { z } { = } 0 . 8 8$ , Wilcoxon $\scriptstyle { p < 0 . 0 0 1 }$ Holm-corrected) and 24/35 on leaveone-motion-out (+5.4 mm, � =0.51, �=0.018). Under MobilePoseradapted the gap closes to a tie on held-out repeats $\scriptstyle ( p = 0 . 9 7 )$ and a non-significant head-favouring trend on held-out motions (�=0.23). The feet-only configuration is the worst trained variant (133.5 mm on held-out runs, worse than feeding no sensor at all), so the foot streams are individually weak and damaging in combination. Discarding the foot IMUs is therefore a validated design decision, not discarded information.

Table 2: Full 35-take benchmark. Distal metrics average knees, ankles, and feet. Lower is better for errors, higher for $R ^ { 2 } .$ “run”/“seq” = leave-one-run-out / leave-one-motion-out. Best per family in bold.
<table><tr><td>Method</td><td>rigid-MPJPE↓</td><td> $R _ { \mathrm { m o t i o n } } ^ { 2 } \uparrow$ </td><td>distal rigid↓</td><td>distal R² ↑</td><td>MPJVE↓</td><td>PA-MPJPE↓</td></tr><tr><td>Constant mean-pose prior</td><td>105.5</td><td>0.03</td><td>141.2</td><td>-0.07</td><td></td><td></td></tr><tr><td>Zero-input control (run)</td><td>116.4</td><td>-0.40</td><td>141.6</td><td>-0.09</td><td>221</td><td>63.9</td></tr><tr><td colspan="7">IMUPoser-adapted</td></tr><tr><td>No calibration, head+feet (run)</td><td>107.0</td><td>-0.18</td><td>130.7</td><td>0.07</td><td>264</td><td>59.0</td></tr><tr><td>Head IMU only (run)</td><td>79.0</td><td>0.06</td><td>87.0</td><td>0.49</td><td>253</td><td>57.3</td></tr><tr><td>Head IMU only (seq)</td><td>92.6</td><td>-0.10</td><td>107.2</td><td>0.27</td><td>249</td><td>64.3</td></tr><tr><td>Head IMU only, causal (run)</td><td>92.9</td><td>-0.09</td><td>107.9</td><td>0.25</td><td>280</td><td>60.5</td></tr><tr><td>Head IMU only, causal (seq)</td><td>91.5</td><td>-0.09</td><td>105.3</td><td>0.28</td><td>259</td><td>61.2</td></tr><tr><td>Feet only (run)</td><td>133.5</td><td>-0.86</td><td>170.5</td><td>-0.98</td><td>267</td><td>56.8</td></tr><tr><td>Feet only (seq)</td><td>129.7</td><td>-0.69</td><td>164.2</td><td>-0.68</td><td>292</td><td>66.8</td></tr><tr><td>Head+feet (run)</td><td>95.8</td><td>-0.05</td><td>114.1</td><td>0.24</td><td>258</td><td>54.6</td></tr><tr><td>Head+feet (seq)</td><td>107.4</td><td>-0.21</td><td>131.6</td><td>-0.01</td><td>271</td><td>62.9</td></tr><tr><td colspan="7">MobilePoser-adapted</td></tr><tr><td>Head IMU only (run)</td><td>85.5</td><td>0.05</td><td>98.7</td><td>0.37</td><td>258</td><td>58.9</td></tr><tr><td>Head IMU only (seq)</td><td>93.7</td><td>-0.04</td><td>111.0</td><td>0.23</td><td>251</td><td>63.1</td></tr><tr><td>Head+feet (run)</td><td>86.9</td><td>-0.02</td><td>97.9</td><td>0.43</td><td>277</td><td>59.2</td></tr><tr><td>Head+feet (seq)</td><td>99.1</td><td>-0.21</td><td>116.3</td><td>0.15</td><td>265</td><td>67.1</td></tr></table>

Why the feet fail: reliability, not placement. Three controls isolate the mechanism. (i) An explicit no-calibration variant with head+feet is worse still (107.0 mm), implicating foot-orientation calibration. (ii) A mounting-bias probe injects a constant random sensor-to-bone rotation per take into the foot streams (per-axis std $\sigma ,$ three draws each). Realistic spread (�=5–10<sup>◦</sup>) is near-neutral on average (92.6–92.8 mm vs. 95.8 mm uninjected) but highly variable across draws (85.0–98.7 mm): individual sessions are luckdependent. Beyond that, degradation is monotonic: 102.4±5.0 mm at $\sigma { = } 2 0 ^ { \circ }$ and 118.9±1.2 mm at $\sigma { = } 4 0 ^ { \circ }$ , past even the 105.5 mm meanpose prior, because the network cannot down-weight a stream it no longer trusts. Even the luckiest draw remains worse than discarding the feet (79.0 mm). (iii) The AirPods stream, by contrast, exposes a clean raw gyroscope, so head angular motion, correlated with stride, turning, and vertical excursions, is directly observed. The Striv insoles provide only a fused Euler orientation with no raw gyroscope, leaving a calibration residual the models cannot overcome. For this consumer device set, sensor reliability, not sensor count, decides the outcome.

Foot contact from the head alone. Foot-contact timing underpins gait segmentation, animation, and clinical timing measures. The auxiliary contact head, driven by the head IMU alone, reaches 0.809 macro-F1 over all 35 held-out takes while preserving pose accuracy (77.0 mm rigid-MPJPE); it is easiest where feet are often stationary (upper-limb 0.968 F1) and hardest in turning (0.694). Temporal foot-state is recoverable from head motion alone even though foot orientation is too unreliable to help pose.

The full-body boundary. Finally we widen the output to a 20-joint full-body skeleton (three seeds, leave-one-run-out). The boundary runs through the arm: the upper-body group beats its prior $( 1 0 5 . 1 \pm 4 . 2 $ vs. 131.5 mm) with distal signal (elbows $R ^ { 2 } { = } 0 . 4 5 { \pm } 0 . 0 6 ,$ wrists $R ^ { 2 } { = } 0 . 3 0 { \pm } 0 . 0 2 )$ tracking gross arm swing, while neck, head, and shoulders sit at or below the prior. This ceiling is not a label artifact: re-scoring held-out predictions against four-view-fused SAM 3D Body labels (per-frame lower-body-anchored similarity fusion, ten-take subset; cross-view residual 10–22 mm) moves proximalupper errors $< 2$ mm. Naive joint training, however, sacrifices the legs to the static-prior level (107.5±13.1 mm vs. the 105.5 mm prior; the dedicated model reaches 79.0 mm). The cost is partly mechanical (unsupervised spine rotations corrupt the kinematic chain; weakly supervising them recovers 137.9 → 115.1 mm) and partly multi-task dilution from the arm labels. Staged fine-tuning (ten lower-only epochs, then ten with the full weighted loss) removes most of the trade-of: the legs recover to 86.2±7.2 mm, within 7 mm of the dedicated model, with the distal-arm signal intact (elbows $R ^ { 2 } { = } 0 . 4 1 )$ Freezing the trunk after the lower stage protects the legs (85.7 mm) but eliminates the arm signal $( R ^ { 2 } \lesssim 0 . 0 5 ;$ single-instance). On unseen motion classes the staged arm signal largely vanishes $\left( R ^ { 2 } { = } 0 . 0 6 \right)$ , and adding the feet hurts again (130.0 mm legs, distal-arm $R ^ { 2 } \lesssim 0 . 2 ) _ { ☉ }$ , so the reliability thesis holds across output sets. One earbud IMU thus carries usable signal for lower-body pose and gross arm swing via staged fine-tuning; a per-task model pair remains accuracy-optimal.

## 7 Discussion and Limitations

Our reading is constructive: head-only is not a compromise but the optimum of the consumer sensor set tested; the foot IMUs served their purpose by being falsified. Though foot sensors carry gaitphase and contact information, in the consumer setting this signal is unusable. We attribute the degradation to the quality and calibration of the consumer insole orientation stream (Sec. 6). The pilot is limited to one subject, one capture day, and SAM 3D Body pseudo-GT rather than marker-based motion capture; our claims therefore establish within-participant feasibility for the tested hardware, not population-level generalization. A label-quality control against four-view-fused SAM 3D Body labels (Sec. 6) confirms single-view label noise does not drive the proximal ceiling, though the fused labels share the same backbone. Next steps: raw-gyroscope insole firmware and calibration, marker-based validation, multi-subject capture, and external baselines.

## 8 Conclusion

We asked how much of the 3D body pose a single consumer earbud IMU can recover, and built the capture pipeline and 35-take benchmark to answer it on real data. The answer: enough for lower-body pose (79.0 mm rigid-MPJPE), foot contact (0.809 macro-F1), and, with staged fine-tuning, gross arm swing, at streaming latency; and never less, often significantly more, than with two extra foot IMUs. For the consumer devices tested, the design question is not sensor count, but which stream is reliable enough to learn from, and here that stream is the head.

## Acknowledgments

This work was supported by a UKRI Future Leaders Fellowship [grant number G127262].

## References

[1] Peng Dai, Yang Zhang, Tao Liu, Zhen Fan, Tianyuan Du, Zhuo Su, Xiaozheng Zheng, and Zeming Li. 2024. HMD-Poser: On-Device Real-time Human Motion Tracking from Scalable Sparse Observations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). arXiv:2403.03561.

[2] Andrea Dittadi, Sebastian Dziadzio, Darren Cosker, Ben Lundell, Thomas J. Cash man, and Jamie Shotton. 2021. Full-Body Motion From a Single Head-Mounted Device: Generating SMPL Poses From Partial Observations. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV). 11687–11697.

[3] Yuming Du, Robin Kips, Albert Pumarola, Sebastian Starke, Ali Thabet, and Artsiom Sanakoyeu. 2023. Avatars Grow Legs: Generating Smooth Human Motion from Sparse Tracking Inputs with Difusion Model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[4] Matic Gregorčič and Dejan Georgiev. 2025. The Usefulness of Wearable Sensors for Detecting Freezing of Gait in Parkinson’s Disease: A Systematic Review. Sensors 25, 16 (2025), 5101.

[5] Ryosuke Hori, Hiroyuki Deguchi, Tsubasa Maruyama, Mitsunori Tada, and Hideo Saito. 2025. Gait Inertial Poser (GIP): Gait-Aware Human Motion Capture Using Shoe-Embedded IMUs. IEEE Access 13 (2025), 183262–183282.

[6] Ryosuke Hori, Jyun-Ting Song, Zhengyi Luo, Jinkun Cao, Soyong Shin, Hideo Saito, and Kris Kitani. 2026. Ground Reaction Inertial Poser (GRIP): Physics-based Human Motion Capture from Sparse IMUs and Insole Pressure Sensors. arXiv preprint arXiv:2603.16233 (2026).

[7] Fuyang Huang, Ailing Zeng, Minhao Liu, Qiuxia Lai, and Qiang Xu. 2020. Deep-Fuse: An IMU-Aware Network for Real-Time 3D Human Pose Estimation from Multi-View Image. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV). 429–438.

[8] Yinghao Huang, Manuel Kaufmann, Emre Aksan, Michael J. Black, Otmar Hilliges, and Gerard Pons-Moll. 2018. Deep Inertial Poser: Learning to Reconstruct Human Pose from Sparse Inertial Measurements in Real Time. ACM Transactions on Graphics (SIGGRAPH Asia) 37, 6 (2018), 185:1–185:15.

[9] Jiaxi Jiang, Paul Streli, Xuchong Luo, Christoph Gebhardt, and Christian Holz. 2024. EgoPoser: Robust Real-Time Egocentric Pose Estimation from Sparse and Intermittent Observations Everywhere. In Proceedings ofthe European Conference on Computer Vision (ECCV). arXiv:2308.06493.

[10] Jiaxi Jiang, Paul Streli, Huajian Qiu, Andreas Fender, Larissa Laich, Patrick Snape, and Christian Holz. 2022. AvatarPoser: Articulated Full-Body Pose Tracking from Sparse Motion Sensing. In Proceedings ofthe European Conference on Computer Vision (ECCV). 443–460.

[11] Yifeng Jiang, Yuting Ye, Deepak Gopinath, Jungdam Won, Alexander W. Winkler, and C. Karen Liu. 2022. Transformer Inertial Poser: Real-time Human Motion Reconstruction from Sparse IMUs with Simultaneous Terrain Generation. ACM Transactions on Graphics (SIGGRAPH Asia) (2022).

[12] Mingyang Li and Anastasios I. Mourikis. 2014. Online temporal calibration for camera–IMU systems: Theory and algorithms. The International Journal of Robotics Research 33, 7 (2014), 947–964.

[13] Matthew Loper, Naureen Mahmood, Javier Romero, Gerard Pons-Moll, and Michael J. Black. 2015. SMPL: A Skinned Multi-Person Linear Model. ACM Transactions on Graphics (SIGGRAPH Asia) 34, 6 (2015), 248:1–248:16.

[14] Naureen Mahmood, Nima Ghorbani, Nikolaus F. Troje, Gerard Pons-Moll, and Michael J. Black. 2019. AMASS: Archive of Motion Capture as Surface Shapes. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV). 5442–5451.

[15] Vimal Mollyn, Riku Arakawa, Mayank Goel, Chris Harrison, and Karan Ahuja. 2023. IMUPoser: Full-Body Pose Estimation using IMUs in Phones, Watches, and Earbuds. In Proceedings of the CHI Conference on Human Factors in Computing Systems. arXiv:2304.12518.

[16] Luke Sy, Nigel H. Lovell, and Stephen J. Redmond. 2021. Estimating Lower Body Kinematics using a Lie Group Constrained Extended Kalman Filter and Reduced IMU Count. IEEE Sensors Journal (2021). arXiv:2103.11393.

[17] Tom Van Wouwe, Seunghwan Lee, Antoine Falisse, Scott Delp, and C. Karen Liu. 2024. DifusionPoser: Real-time Human Motion Reconstruction From Arbitrary Sparse Sensors Using Autoregressive Difusion. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

[18] Haolong Wang, Zihao Yang, Hao Wang, and Lin Wang. 2025. Ear2Pos: A Dual-IMU Framework for Full-Body Pose Estimation Using Earbuds. IEEE Internet of Things Journal 12, 15 (2025), 31289–31301.

[19] Vasco Xu, Chenfeng Gao, Henry Hofmann, and Karan Ahuja. 2024. MobilePoser: Real-Time Full-Body Pose Estimation and 3D Human Translation from IMUs in Mobile Consumer Devices. In Proceedings of the ACM Symposium on User Interface Software and Technology (UIST). arXiv:2504.12492.

[20] Dongseok Yang, Jiho Kang, Lingni Ma, Joseph Greer, Yuting Ye, and Sung-Hee Lee. 2024. DivaTrack: Diverse Bodies and Motions from Acceleration-Enhanced Three-Point Trackers. Computer Graphics Forum (Eurographics) (2024). arXiv:2402.09211.

[21] Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt Feiszli, Jitendra Malik, Piotr Dollar, and Kris Kitani. 2026. SAM 3D Body: Robust Full-Body Human Mesh Recovery. arXiv:2602.15989 [cs.CV] arXiv:2602.15989; code and checkpoints at https://github.com/facebookresearch/sam-3d-body.

[22] Xinyu Yi, Yuxiao Zhou, Marc Habermann, Soshi Shimada, Vladislav Golyanik, Christian Theobalt, and Feng Xu. 2022. Physical Inertial Poser (PIP): Physicsaware Real-time Human Motion Tracking from Sparse Inertial Sensors. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 13167–13178.

[23] Xinyu Yi, Yuxiao Zhou, and Feng Xu. 2021. TransPose: Real-time 3D Human Translation and Pose Estimation with Six Inertial Sensors. ACM Transactions on Graphics (SIGGRAPH) 40, 4 (2021), 86:1–86:13

[24] Yu Zhang, Songpengcheng Xia, Lei Chu, Jiarui Yang, Qi Wu, and Ling Pei. 2024. Dynamic Inertial Poser (DynaIP): Part-Based Motion Dynamics Learning for Enhanced Human Pose Estimation with Sparse Inertial Sensors. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 1889–1899. arXiv:2312.02196.

[25] Zunjie Zhu, Yan Zhao, Yihan Hu, Guoxiang Wang, Hai Qiu, Bolun Zheng, Chenggang Yan, and Feng Xu. 2025. Progressive Inertial Poser: Progressive Real-Time Kinematic Chain Estimation for 3D Full-Body Pose from Three IMU Sensors. arXiv preprint arXiv:2505.05336 (2025).