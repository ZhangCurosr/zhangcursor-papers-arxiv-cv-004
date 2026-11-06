# VisionSIM: Beyond Idealized Sensors — A Multi-Modal Simulator for Robot Perception

Sacha Jungerman<sup>1</sup> Kaustubh Sadekar<sup>2</sup> Avery Gump<sup>1</sup> Sriram Narayanan<sup>3</sup>

Shantanu Gupta<sup>1</sup> Bhuyashi Deka<sup>1</sup> Srinivasa Narasimhan<sup>3</sup> Atul Ingle<sup>2</sup> Mohit Gupta<sup>1</sup>

<sup>1</sup>University of Wisconsin – Madison <sup>2</sup>Portland State University <sup>3</sup>Carnegie Mellon University

Abstract: Robotic perception increasingly relies on heterogeneous sensor suites combining cameras with emerging modalities (event, SPAD, iToF/dToF, and thermal). Yet existing simulators idealize these sensors, hiding critical modalityspecific failure modes that limit real-world deployment. We present VisionSIM, a unified, physically grounded simulation platform that elevates sensor realism to a first-class concern by modeling the raw physical measurement process for diverse conventional and emerging modalities. We expose fine-grained control over physical sensor parameters, reproducing failure modes overlooked by general-purpose simulators: pileup and glare for LiDAR; phase wrapping, saturation, and motion artifacts for iToF; refractory drops for event cameras; and history-dependent emission for thermal cameras. This enables principled cross-modality comparison on identical scenes and trajectories. We release the platform and a large indoor dataset with synchronized raw measurements and rich ground-truth annotations.

Keywords: Sensor Simulation, Sim-to-Real, Multi-Modal Sensing, Robotics

## 1 Introduction

A robotics simulator presented with a scene will typically render a sharp, noise-free image. Yet, the same scene captured by a real conventional camera might show heavy motion blur from a long exposure; a real single-photon LiDAR aimed at a retroreflective object might report a phantom obstacle in front of it; a long-wave infrared (LWIR) thermal camera will show a lingering thermal footprint of a warm handprint on a desk or a radiator still glowing minutes after shutoff. These are not exotic edge cases — they are the operating conditions that decide whether a deployed robot succeeds or fails. However, they are largely absent from today’s general-purpose robotics simulators.

Modern simulation platforms have transformed how robots are trained, evaluated, and deployed: physics engines such as MuJoCo [1], Isaac Gym [2], and Genesis [3] model rigid-body dynamics with impressive fidelity, while rendering engines such as Unreal [4] and Blender [5] bring photorealistic digital twins of factories and cities within reach. However, these platforms share a critical blind spot: they do not faithfully simulate how agents actually see the world. Real-world agents act on what their sensors deliver, not on ground truth, and typically combine multiple complementary sensors (Table 1). Yet most simulation engines do not support the necessary breadth of sensors, or they model them only superficially: ideal cameras in a well-lit environment, LiDARs as noiseless range-finders, IMUs as perfect integrators, thermal cameras as static temperature textures. Such simplifications ignore the noise, motion blur, and modality-specific artifacts that dominate real measurements — from glare and pileup in single-photon LiDAR [6] to the geometry-, material-, and history-dependent emission that defines real thermal imagery.

Existing efforts address this gap only partially. Sensor-specific simulators, such as ESIM [7] and v2e [8] for event cameras, are highly specialized and cannot support cross-modality comparisons. Physically-based infrared generators such as MuSES [9] and DIRSIG [10] couple heat-transfer solvers to radiometric rendering, but target terrain-scale signature prediction driven by weather and atmospheric models, rather than the object-scale thermal phenomena central to embodied perception. General-purpose platforms such as CARLA [11], AirSim [12], Isaac Gym [2], and Habitat [13] expand sensor coverage but inherit the same idealizations. As a result, robot system architects cannot compare candidate sensor suites on common ground without expensive hardware prototyping.

<table><tr><td>Modality</td><td></td><td>Raw Measurement</td><td>Recoverable Scene Properties</td><td></td><td>Sim. HW Outlook</td><td></td></tr><tr><td></td><td>RGB cameras*</td><td>Visible radiance</td><td>Appearance, texture, coarse geometry cues</td><td></td><td></td><td></td></tr><tr><td></td><td>Stereo / multi-view*</td><td>Multi-view radiance, parallax</td><td>Metric 3D geometry, surface layout</td><td></td><td></td><td></td></tr><tr><td></td><td>LiDAR (SPAD)*</td><td>Photon arrival times; transients</td><td>Low-light depth, long-range geometry</td><td></td><td></td><td></td></tr><tr><td></td><td>LiDAR (iToF)*</td><td>Correlation measurements</td><td>Depth, dynamic structure</td><td></td><td></td><td></td></tr><tr><td></td><td>Radar (mmWave)</td><td>RF phase, Doppler, reflections</td><td>Coarse geometry, velocity, micro-motion</td><td></td><td></td><td></td></tr><tr><td></td><td>Thermal (LWIR)*</td><td>Thermal emission (8–14 µm)</td><td>Temperature fields, heat signatures</td><td></td><td></td><td></td></tr><tr><td></td><td>Event cameras*</td><td>Asynchronous intensity changes</td><td>High-speed motion, edges, dynamics</td><td></td><td></td><td></td></tr><tr><td></td><td>Inertial Unit*</td><td>Forces, acceleration, joint states</td><td>Contact state, pose, inertial dynamics</td><td></td><td></td><td></td></tr><tr><td></td><td>Audio / acoustic</td><td>Pressure waves and echoes</td><td>Occluded structure, material &amp; cavity cues</td><td>☆</td><td></td><td></td></tr><tr><td></td><td>Hyperspectral</td><td>Spectral reflectance signatures</td><td>Material, chemical, biological composition</td><td></td><td></td><td></td></tr></table>

Adoption: Established Ramping Up Fringe Maturity: ⋆ Low . Med ⋆ High Outlook: 5 stable  growth m accelerating  
Table 1: Computational imaging modality taxonomy. Each modality is characterized by its raw physical measurement and the recoverable scene properties. The Sim. and HW columns indicate simulation and hardware maturity, respectively. Outlook glyphs indicate likely near- to mid-term trajectories in availability and adoption. \*Currently supported by VisionSIM.

Our guiding hypothesis is that a high-fidelity simulator must model not just world physics and geometry, but also perception itself — with realistic sensors that bridge photons to measurement. To this end, we present VisionSIM, a unified, physically grounded sensor simulation platform for robotics that aims to elevate sensor realism to a first-class concern, on par with scene and physics modeling (Figure 1). VisionSIM is built around the following key design goals:

Breadth across conventional and emerging sensors. VisionSIM simulates conventional and highspeed cameras, stereo rigs, passive single-photon avalanche diode (SPAD) cameras, direct and indirect time-of-flight (dToF/iToF) LiDARs, event cameras, long-wave infrared (LWIR) thermal cameras, and inertial measurement units (Table 1). Candidate sensor suites can be evaluated on identical scenes and trajectories, enabling principled comparisons across a wide range of sensors.

High physical fidelity, including failure modes. VisionSIM emulates raw measurements for each modality (e.g., transient histograms, correlation taps, and asynchronous brightness changes). This naturally reproduces key failure modes: SPAD pileup and glare; iToF wrapping and motion artifacts; event refractory limits; and path-tracing-coupled dynamic thermal heat transport, capturing failure modes that, to our knowledge, lie beyond the reach of existing general-purpose simulators.

Fine-grained control over sensing parameters. VisionSIM exposes the physical parameters that govern each modality’s measurement process (Table 2): iToF correlation functions; event contrast thresholds; dead time and quantum efficiency for SPADs; microbolometer thermal time constants, and per-surface emissivity for thermal cameras; exposure, gain, and ISP parameters for conventional cameras. This parameterization lets a user not only sweep across sensor designs, but also match the simulator to a specific physical device — the basis for calibrated perceptual digital twins.

We demonstrate VisionSIM by constructing a digital twin of a real single-photon LiDAR, recovering some of its key parameters directly from captured data. This reduces the mean sim-to-real depth error and reproduces two failure modes not previously simulated at this fidelity: pileup-induced range walk, and point-cloud ghosting from retroreflective glare. We further show how VisionSIM can be used to augment existing datasets and to evaluate competing sensor suites (Sec. 5). These case studies focus on open-loop sensor fidelity; closed-loop validation is discussed in Section 8. Finally, we release an indoor dataset featuring synchronized raw outputs of all supported sensors and ground truth annotations (Table 3). To our knowledge, it is among the most comprehensive multi-modal datasets of its kind, supporting both training and benchmarking of sensor-fusion algorithms.

## 2 Related Works

Task Specific Datasets & Pipelines: There exist several publicly available real and synthetic datasets supporting various computer vision tasks. These are often specific to applications, such as autonomous driving [14, 15, 16, 17], or tasks, such as optical flow or stereo matching [18, 19, 20, 21]. They often lack the metadata, ground truth, and raw sensor access needed for use beyond their original task. The ones with rich annotations are either subsets of larger datasets [15, 16], or are synthetically generated using purpose-built engines such as CARLA [11], which are not easily generalizable to other tasks and may lack unconventional modalities such as single-photon or thermal sensors.

Thermal  
![](images/e397838df78c27c71552c3a66d00a2b33853d4bef86852aa54ca2beee1a3c9f8.jpg)

![](images/d7d0b6ce25c148b0063a496551b2130d9b1c416d5acef665810f4561c3ae4280.jpg)

![](images/66558a93b412d9a0c39f3604bd48d959c2f5341767b017db89b4a9014354acfe.jpg)

![](images/8991f232c9e8df4193638728a91f90b8667b702c26dff873fdfc7a68b31edcec.jpg)

![](images/20f3d2ec22abdc8dce04ef10609d666c541c18d2cfa17ca047450347676a08c4.jpg)

![](images/467c1e532974262efe4e8903f35b1006c0dceb894750ccf564b1afbc8b76cd59.jpg)

![](images/0ce14bbcb5da2be2bb3ea3fba54acc65ebe78bfb6eaf709e49bd009ab878e8bf.jpg)

![](images/a05f5d0141a5c924df9974f19e4486dd01750e6ce9d0ffbbc45913d193f67600.jpg)

![](images/4c874d1b49832886f78fdcea8bdee83ff4a7e7aa12c4485e01dd5923ab122890.jpg)

![](images/a3f617b3484ace80c86fd30682fb220320d3401a80b33cd5bccc9843e5c76302.jpg)

![](images/f6465d974b42bd471702b1cac36e651510a4081052b2543d6a0df954fde7d958.jpg)

![](images/58d77f7c385cadd297511e5ddf67a02d3bacc0f5448473c6a5c4bcc88c033b91.jpg)

![](images/d99738893ebd41c2132ef76672c29d50f80d33d8b88b9a906444f4fccd0720de.jpg)

![](images/8ba8343da4288d146009614def685e983176559749c7a8d2b8f1153eda2740da.jpg)

![](images/4de9d7988c5f9bc99bd9206784c60bd2f3813d1d4bf4080d03c7a254ae25e3d5.jpg)

![](images/5f838a01f776680c296a023733df94cdb60e608742a59601fb2975374e86bf67.jpg)  
Figure 1: Multi-modal, high-fidelity and controllable sensor emulation. Robotic simulation pipelines often focus on scene content, overlooking physically accurate sensor modeling. VisionSIM aims to bridge this gap by emulating diverse sensor types and their failure modes. (Top) A single scene is rendered once with dense ground truth — color, depth, optical flow, surface normals, segmentation, point map, direct and indirect illumination, and a temperature map. (Middle) The same scene is then replayed through a breadth of sensor models. (Bottom) Each model exposes and enables sweeping across the sensor’s physical parameters (e.g., thermal time constant, correlation function, event thresholds), thereby reproducing modality-specific failure modes.

General Purpose Simulation Engines: Many engines (Isaac Gym, Genesis, PhysX, Kubric, Bullet) [2, 3, 22, 23, 24] prioritize fast and accurate physics for robotics learning, often at the cost of rendering fidelity. Frameworks such as BlendProc [25] and Kubric [23] yield photorealistic results, yet perform no sensor emulation. Others, such as ESIM [7], target a single sensor type. VisionSIM addresses both aspects — sensor breadth with physical realism — within a unified pipeline.

Sensor Emulation Methods: Emulating different sensor types or characteristics from existing real data is an active research direction, lowering the barrier to adopting non-standard sensors. v2e [8] converts videos to event streams; physics-based infrared generators such as DIRSIG [10] and MuSES [9] couple heat solvers to rendering, but are proprietary or domain-specific. Vision-SIM combines these trends by bringing single-photon [26, 27, 28], transient, and light-driven ther mal [29, 30] sensing into a unified, multi-modal simulation framework for embodied perception.

## 3 Architecture Overview

VisionSIM’s key design choice is to decouple photorealistic rendering from sensor emulation. We first render the clean scene, and then apply physical non-idealities to produce raw measurements. Because these layers are separate, one rendered sequence can be replayed through any sensor model, and parameters can be tweaked at zero extra rendering cost, enabling rapid device calibration.

World Simulation Layer. This layer is the image-formation engine, producing photorealistic reference frames together with dense ground-truth annotations (depth, normals, optical flow, segmentation, point maps, illumination, poses, and intrinsics) from a Blender scene. It additionally generates ground-truth temperature fields through an integrated, GPU-accelerated heat-transport solver that drives the thermal sensor model (Appendix S.5.4). A central challenge is producing frames at the extreme rates demanded by high-speed and asynchronous sensors — event and single-photon cameras operate at tens to hundreds of thousands of frames per second — which is prohibitively expensive to render. VisionSIM addresses this with distributed client/server rendering across heterogeneous hardware, a keyframe-remapping technique that temporally dilates animation splines so complex physics can be sampled at arbitrary frame rates, and a render-then-interpolate strategy that renders ground-truth frames only at each scene’s intrinsic nativeframe rate and synthesizes the rest with deep video interpolation (e.g., RIFE [31]). Together, these yield order-of-magnitude reductions in wall-clock time while preserving high fidelity to fully rendered sequences (Appendix S.2-S.2.7).

<table><tr><td>Modality</td><td>Modeled Non-idealities &amp; Failure Modes</td><td>Key Exposed Parameters</td></tr><tr><td>o RGB cameras</td><td>Shot/read noise, motion blur,</td><td>Exposure, gain, full-well capacity,</td></tr><tr><td>High-speed cameras</td><td>full-well saturation, quantization Elevated read noise, low signal, motion blur</td><td>bit depth, ISP artifacts Exposure, gain, read noise</td></tr><tr><td>Stereo / multi-view</td><td>Disparity error, inherited camera non-idealities</td><td>Baseline, intrinsics, per-camera parameters</td></tr><tr><td>業</td><td>LiDAR (SPAD, dToF) Pileup, range walk, count saturation, ambient background, glare</td><td>IRF, dead time, quantum efficiency, ambient rate, TDC resolution</td></tr><tr><td>LiDAR (iToF)</td><td>Phase wrapping, saturation, ambient corruption,</td><td>Coding function, modulation frequency,</td></tr><tr><td>Passive single-photon</td><td>multi-tap &amp; intra-tap motion, noise, quantization</td><td>taps K, full-well capacity, read noise, ADC bits</td></tr><tr><td>Thermal (LWIR)</td><td>Shot-noise, quantum efficiency, dark counts Temperature-dependent emission, thermal inertia</td><td>Exposure, dark count rate, fill factor</td></tr><tr><td>Event cameras</td><td></td><td>Thermal parameters, FEM simulation</td></tr><tr><td>IMU</td><td>threshold variation, refractory drop-out, shot noise Random-walk drift, bias instability, Gaussian noise Noise, bias instability, output rate</td><td>Contrast threshold, refractory period</td></tr></table>

Table 2: Failure modes and parameters modeled by VisionSIM. For each supported modality, we list the non-idealities and failure modes VisionSIM reproduces and the principal user-controllable parameters. Full forward models for all modalities are given in Appendix S.3.

Sensor Emulation Layer. This layer ingests the noise-free, high-frame-rate data and transforms them into realistic sensor measurements. Rather than applying generic additive Gaussian noise to idealized outputs, it implements physically constrained, per-modality measurement models from which noise, distortions, and failure modes naturally emerge as physical consequences. The layer supports conventional and high-speed cameras, stereo and multi-camera rigs, passive single-photon cameras, single-photon (dToF) LiDAR, indirect time-of-flight (iToF) LiDAR, event cameras, thermal cameras, and inertial measurement units. Since it operates on shared reference frames, any subset can be co-simulated on identical scenes and trajectories. The resulting multi-modal measurements are saved in a compressed, sensor-aware format compatible with downstream learning frameworks [32, 33]; stochastic binary single-photon frames, in particular, are stored in a bit-packed manner that achieves 8× lossless compression with on-the-fly random access (Appendix S.4).

## 4 Physically-Grounded Sensor Models and Failure Modes

We now detail the physically accurate sensor models for three representative modalities whose failure modes are least faithfully captured by existing simulators: single-photon LiDAR, indirect timeof-flight LiDAR, and thermal cameras. Forward models and implementation details for all supported sensors are in Appendix S.3; Table 2 summarizes their failure modes and parameters.

## 4.1 Single-Photon (dToF) LiDAR

Commercial solid-state LiDAR is rapidly adopting single-photon avalanche diode (SPAD) arrays [34, 35] to capture photon-arrival histograms for depth estimation. VisionSIM physically models the histogram acquisition pipeline. For each pixel, the discrete scene response $T [ b ]$ at bin index b aggregates the radiometric contributions of scene points within a field-of-view mask M as

$$
T [ b ] = \eta _ { \mathrm { o p t } } \sum _ { p \in \mathcal { M } } \mathbb { I } \big ( b \Delta t \leq t _ { p } < ( b + 1 ) \Delta t \big ) \frac { \alpha _ { p } } { z _ { p } ^ { 2 } } W ( p ) ,\tag{1}
$$

where $\Delta t$ is the temporal bin width, $\alpha _ { p }$ is albedo, $1 / z _ { p } ^ { 2 }$ the inverse-square falloff, $W ( p )$ the mask weight, and $\eta _ { \mathrm { o p t } }$ the quantum efficiency. The spatially-resolved mask M models common optical effects like vignetting and blooming from retroreflective returns (Section 5). SPAD crosstalk is not yet modeled; it can be added as a per-pixel spatial kernel applied to the transient, which would capture optical and electrical coupling between neighbouring detectors. The total photon-arrival rate is then modeled as $\lambda ( t ) = ( T * g ) ( t ) + \lambda _ { \mathrm { a m b i e n t } }$ , where $\lambda _ { \mathrm { a m b i e n t } }$ is a constant background rate, and $g ( t )$ is the system’s impulse response function (IRF), a calibrated or parametric laser pulse. Finally, incident photon counts are drawn from an inhomogeneous Poisson process governed by λ(t).

![](images/c7042386a1341539c58990004f4acde1797fb569bdc30230055bad798d6f7427.jpg)  
Figure 2: Indirect Time-of-Flight (iToF) simulation and failure cases: (Top) Ground truth RGB scene, ground truth depth map, and ideal correlation measurements (taps 1, 2, and 3) for a conventional sinusoidal coding scheme. (Bottom) Common iToF failure cases and their effects on depth reconstruction, including: incorrect modulation frequency (causing phase wrapping), sensor saturation (clipping correlation measurements), ambient light corruption (reducing signal-to-noise ratio, especially for farther objects), motion blur (multi-tap artifacts from scene dynamics), and sensor noise and quantization artifacts (high frequency depth errors).

In high-flux conditions, modeling dead time (time interval after a photon detection when the sensor is inhibited) emulates pileup-distorted measurements [36, 37]: later-arriving photons are systematically lost, resulting in histograms biased toward earlier bins and shallower depth estimates (range walk error) and poor SNR, particularly for distant objects (Fig. 1). Intra-histogram motion blur can cause erroneous depth estimates, causing safety-critical issues (Fig. 4). Conventional simulators that model LiDARs as noiseless range-finders would fail to capture these effects.

## 4.2 Indirect Time-of-Flight (iToF) LiDAR

Indirect, or continuous-wave, time-of-flight cameras are similarly widespread, shipping in consumer and industrial depth sensors from major manufacturers [38, 39]. Rather than timing individual photons, they recover depth from the phase of amplitude-modulated light [40]: the scene is illumi nated with a periodically modulated source, and the sensor correlates the returning light against K phase-shifted demodulation functions, producing K correlation taps from which depth is decoded. Following the continuous-wave image-formation model [41], the i-th measured tap at a pixel is:

$$
\tilde { B } _ { i } = \mathcal { Q } ( \beta F _ { i } ( \Gamma ) + A + \eta _ { i } ) , \quad i = 1 , \dotsc , K ,\tag{2}
$$

where Γ is the scene depth, $F _ { i }$ the normalized correlation function for tap i (set by the coding scheme), $\beta$ includes albedo and distance-squared falloff, A is the ambient component, $\eta _ { i }$ is shot and read noise, and $\mathcal { Q } ( \cdot )$ models sensing non-idealities (e.g., ADC quantization, saturation).

Equation 2 captures physical failure modes explicitly (Figure 2): (i) Phase wrapping: depth wraps beyond the unambiguous range, $\Gamma _ { \mathrm { m a x } } = c / 2 f$ , set by the modulation frequency $f ,$ rendering distant objects close; (ii) Saturation: measurements clip when signal exceeds full-well capacity; (iii) Ambient corruption: large A reduces modulation contrast, corrupting depth particularly for distant, dimly-lit points; (iv) Motion artifacts: intra- and inter-tap motion (for sequential taps) cause spatial tearing and boundary ghosting, which we model by integrating temporally offset interpolated frames; (v) Noise $\&$ quantization: $\eta _ { i }$ and $\mathcal { Q }$ cause high-frequency depth errors and banding.

## 4.3 Thermal Cameras

Long-wave infrared (LWIR) thermal cameras differ from other modalities in that they not only measure reflected thermal light due to light transport but also capture light emitted by an object due to the underlying heat transport process happening within and around the object. The radiance at each pixel is the integrated long-wave emission of the surface in front of it, governed by Planck’s law and weighted by the surface’s emissivity, with an additive reflected-ambient term.

![](images/799433554d93111fabcc9116e809972c6d28f7c0300661fb434ed2575db46b3d.jpg)  
Figure 3: Transient Thermal Simulation: Thermal video of ceramic, steel, and aluminum cups over the 4 seconds following a hot-tea fill. The distinct heating rates reflect each material’s thermal diffusivity (α = 2, 18.8, 97 mm<sup>2</sup>/s, respectively).

Crucially, this signal depends not on instantaneous lighting but on the temperature field that geometry, materials, and the scene’s recent history (minutes to hours of prior illumination, contact, and conduction) have produced — a quantity conventional graphics pipelines do not compute. Existing thermal-simulation tools [12] sidestep this by painting a static “temperature texture” onto each surface or assuming radiative equilibrium with a single user-provided heat source [42]; neither approach reproduces the material- and

history-dependent thermal contrast that defines real LWIR imagery, such as differential cooling across materials (ceramic vs. metal mugs cooling at different rates in Fig. 3), residual heat from recent contact, or radiator-like sources that linger after shutoff. VisionSIM directly integrates a fast physics-based heat solver in its world simulation layer, so thermal imagery can be produced alongside the other modalities. For details on our light-to-heat simulation, thermal rendering, and microbolometer inertia, see Appendix S.5.4.

## 5 Case Studies: Design and Analysis of Robot Perception Systems

Next, we illustrate potential applications of VisionSIM, each with a case study (Figure 4) designed to serve as a proof-of-concept via a representative example. All studies are open-loop, focusing on sensor fidelity rather than closed-loop control (Section 8).

Building digital twins of real sensors. VisionSIM enables building perceptual digital twins: a model whose parameters are calibrated so that its simulated raw measurements match those of a specific physical device, capturing the sensor’s characteristic noise and failure modes. This is critical for closing the sim-to-real gap on the target deployment hardware.

We build a digital twin of a commercial single-photon LiDAR (Adaps Photonics ADS6311 [43]; details in Appendix S.6). Because VisionSIM’s dToF model is differentiable, we recover the device’s parameters directly from real captures via gradient-based fitting of a measurement-matching loss, which we demonstrate with the sensor’s impulse response function (IRF). We validate the twin on a scene captured in the lab and modeled in Blender (Figure 4, top), chosen because it exercises two failure modes at once. The retroreflective sign produces an intense return that saturates the detector; the recovered IRF and dead time reproduce the resulting pileup and range-walk error. That same intense return induces glare through internal multipath between the sensor optics and the detector array, which the spatially-varying PSF captures, reproducing the characteristic point-cloud ghosting. The resulting reconstructed depth closely tracks the device’s measured depth. Faithfully modeling the IRF, pileup, and glare lowers the RMSE of the digital twin by ∼1.53 meters.

Migrating existing datasets to new sensing modalities. The decoupled architecture also enables migrating existing datasets to modalities they were never captured with, by bypassing world simulation and feeding pre-existing data directly to the sensor emulation layer. We show three examples (Figure 4, middle): emulating a passive single-photon camera from the XVFI high-speed video dataset [44]; generating an event stream from a Genesis render of a dancing humanoid [3, 45]; and producing raw single-photon LiDAR transients from a CARLA driving sequence [11]. Uplifting a conventional autonomous-driving dataset into time-resolved single-photon measurements lets sensor-specific failure modes be exposed: here, the pedestrian might not be detected at the sensor level as their lateral motion induces motion blur within the transient histograms, causing depth errors in excess of 3 meters. This lets researchers obtain data for emerging sensors by reusing existing assets, rather than recapturing scenes with specialized and often unavailable hardware. Dataset migration is bounded by what the source data contains: the frame rate caps the temporal resolution of the emulated sensor, and each modality needs its own annotations, so a dataset without depth cannot be uplifted to iToF or dToF.

Captured Depth  
Single-Photon Data from Video  
Digital Twin of real-world sensors  
![](images/0716f650eb7fda0f0207d518d51e4efceb1a25ed7f2d41cc68f6d9d69b19509f.jpg)  
Sensor Modality Migration

![](images/153c64bdc1adc7f1c452e4ea41e015cdc3cd949c7829473ff0c0eab13a1029c7.jpg)

![](images/be23024dbb15a1a6fcd1f36f8ce6660b3596ca8839a25560e1cdbaf977db86e1.jpg)

![](images/2c83af5f0b9546e6d2f1a210245caaf691b82cdd821cbe3649e63b6b3df5460e.jpg)  
Event Stream from Physics Engine

![](images/10b7a205c8f65fa04735f1e20161f55d3743020059ee8e1d60ad5d1c0852be89.jpg)

![](images/05c28164607829c2442bd298968ca6d6cb11120df412a18ff9304dc9a04d1de9.jpg)

Realistic LiDAR from Autonomous Vehicle Simulation  
Sensor Suite Evaluation  
![](images/876bab2016b86aa3545e9acbe69fce2e2cc075097a4008cb69679b8feda0688e.jpg)  
Figure 4: Application Case-Studies: (Top) A digital twin of a commercial LiDAR, where sensor parameters such as dead time and impulse response function are estimated from real-world data to correct for artifacts like range-walk. (Middle) Sensor modality migrations: VisionSIM can uplift existing datasets (e.g., XVFI [44]) or high-speed renders to emulate new sensor modalities like single-photon or event cameras. (Bottom) Sensor suite evaluation: VisionSIM enables rapid evaluation of different sensor configurations for perception tasks.

Evaluating & benchmarking sensor suites. Sensor emulation decoupling enables a single rendered scene to be replayed through many sensor configurations at no additional rendering cost, allowing comparisons of candidate suites before any hardware is built. We demonstrate this through a robot perception case study, where a system architect selects between two sensor suites for lowlight, dynamic environments: a conventional camera paired with an event camera, or a passive single-photon camera and IMU. Both are emulated on identical high-speed trajectories; each suite’s raw measurements are reconstructed into image-like representations using existing methods [46, 27], and passed to an off-the-shelf detection and tracking model [47] to compare downstream task performance (Figure 4, bottom). Detection fails in the first case, while the second sensor suite yields 90% confidence. Both reconstruction models and the detector are used zero-shot, so the result characterizes the full pipeline rather than the sensor suite alone; retraining on raw measurements would remove the reconstruction bias. Sweeping sensor parameters (exposure, event threshold, fill factor, etc.) within the same framework lets the architect map the trade-off across operating regimes, a comparison that is prohibitively expensive to perform with real hardware.

## 6 Sim-to-Real Validation

In the previous section, we calibrated a dToF sensor twin using a handful of real measurements. Here, in Fig. 5, we further compare real and simulated sensor measurements for four additional modalities. The first two rows show sensor measurements simulated end-to-end using a digital twin

Simulated

of a real scene, and compared against captures of the same physical scene. The latter two emulate the simulated measurements using real conventional video, also validating our emulation protocol.

For a mug filling with hot tea, the simulated heating profile tracks a real-world capture from a FLIR Boson to within a mean absolute error of 7.6% of the real mug’s heating dynamic range over the 4 seconds following the fill. Against a real capture of a STOP sign taken with a Chronoptics iToF camera, our emulated iToF sensor achieves a mean phase error of 0.13 radians across a [0, 2π] range. At a 40 MHz modulation frequency, this translates to a mean depth error of about 5.3 cm. Event counts emulated on RGB video of an auditorium differ from a real recording taken with a Prophesee EVK4 sensor by 1.5% for positive events and 5.8% for negative events. The photon detection statistics of our emulated Pi-Imaging SPAD sensor, likewise driven by RGB video, reach 0.221 against a measured 0.229.

Small discrepancies remain in each case. The thermal error follows the emitters surface emissivity and the way our heattransport solver couples the mug to its sur-

![](images/f95ba6aa02ad23008d26093f9ffd1ac8b3bb2dc35177e98a434f77751e5f043b.jpg)  
Figure 5: Simulation Fidelity. Comparison of simulated vs. captured scenes. The top two rows are simulated end-to-end using a digital twin of the real scene, while the bottom two emulate the corresponding sensors from a co-located conventional camera capture (inset).

roundings: since the mug exchanges heat with a room whose ambient temperature we do not measure, a small error in either term propagates into the simulated surface temperature. For iToF, some of the discrepancy comes from the sensor’s departure from the ideal. We assume a sinusoidal coding scheme, so the correlation model misses the multipath and higher-harmonic distortion that the physical Chronoptics device introduces. For the two modalities emulated from RGB video, discrepancies can arise before the sensor model runs. Our event-camera model matches the expected spread of contrast thresholds, but its front-end bandwidth and refractory period use the generic DVS model rather than the EVK4, potentially affecting event counts. The Pi-Imaging emulation faces a tougher challenge: mapping linearly rectified intensities to photon detection statistics requires knowing the source camera’s unknown white balance and uncalibrated photometric response. As a result, the single-photon outcome is limited by the source camera, not our sensor model—an error that does not affect the digital-twin rows of Fig. 5, which are simulated end-to-end.

## 7 The VisionSIM Multi-Modal Sensor Dataset

A key capability of VisionSIM is multi-modal dataset generation: a user selects scenes, camera trajectories, resolution and frame rate, the sensors to emulate, and their physical parameters, and the platform produces synchronized multi-sensor measurements with dense ground truth for a given set of tasks. To demonstrate this capability, we generate and release one such dataset, described below.

An indoor instantiation. We curate 100 unique indoor scenes spanning offices, kitchens, bathrooms, etc., each with a predefined 24-second camera trajectory. Each scene is rendered at 100fps in HDR at 1000 ×1000 as a stereo pair, with ground-truth poses, intrinsics, depth, surface normals, optical and scene flow, segmentation, temperature, and direct/indirect illumination components, yielding 480,000 synchronized ground-truth views. From these, we emulate the full sensor suite: a single-photon camera at 16 kHz (via 160× interpolation), a 4 kHz event camera (40×), a 25 fps conventional camera, dToF and iToF LiDAR, a thermal sensor, an IMU at 250 Hz, and a stereo rig, using representative noise parameters drawn from the literature [48, 49, 50, 8, 51, 52, 53, 19, 6, 41, 54].

<table><tr><td colspan="2"></td><td colspan="4">Annotations</td><td colspan="7">Sensors</td></tr><tr><td>Dataset</td><td>Frames</td><td>Depth</td><td>Flow</td><td></td><td>Sem. Pt. Maps</td><td>Camera</td><td>SPAD</td><td>Event</td><td>dToF</td><td>iToF</td><td>Therm.</td><td>IMU</td></tr><tr><td>nuScenes [14]</td><td>1.4M</td><td>☆</td><td></td><td>★</td><td>★</td><td>四</td><td></td><td></td><td></td><td></td><td></td><td>V</td></tr><tr><td>SPICE-HL3 [55]</td><td>1.2M</td><td></td><td></td><td></td><td></td><td>四</td><td>✓</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CroMo [56] </td><td>80k</td><td>☆</td><td></td><td></td><td></td><td>四</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPD [54] </td><td>103k</td><td></td><td></td><td>☆</td><td></td><td>à</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CARLA-Loc [57]</td><td>66k*</td><td>★</td><td></td><td>★</td><td>★</td><td>四</td><td></td><td>四</td><td></td><td></td><td></td><td></td></tr><tr><td>TartanAir [58] </td><td>1M</td><td></td><td>☆</td><td>★</td><td>☆</td><td>四</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MSE [59]</td><td>100k</td><td></td><td></td><td>★</td><td></td><td>à</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ThermVision-DB [60]</td><td>30k</td><td></td><td></td><td>☆</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FLAT [61] </td><td>2k</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisionSIM (Ours)</td><td>480k</td><td>★</td><td>★</td><td>★</td><td>★</td><td>四</td><td></td><td>四</td><td></td><td></td><td>V</td><td></td></tr></table>

Real Synthetic Scene: Ñ Indoor  Outdoor = Space Streams: \*Unsynchronized Fidelity: ⋆ High . Med ⋆ Low Views: ì Multi-view  Single View Format: Å Point Cloud ¡ Transient Histograms  
Table 3: Landscape of computer vision datasets. Existing datasets focus primarily on specific modalities (e.g., event cameras or LiDAR) or lack advanced sensor emulation (e.g., SPADs or thermal), VisionSIM provides a comprehensive suite of multi-modal data including HDR stereo views, optical flow, point maps, and a wide variety of physically-based sensor emulations in a single unified dataset.

Simulating these streams guarantees perfect spatial co-registration and temporal synchronization—conditions that are exceedingly difficult to meet on a physical rig. Moreover, specialized hardware like high-resolution SPADs remains rare and expensive. As Table 3 shows, our dataset provides broader modality and annotation coverage than prior datasets, with raw sensor data (e.g., transient histograms) rather than just processed outputs.

Release. We release the ground-truth components together with the scripts, assets, and checksums needed to regenerate the full sensor data (due to the large data size). The same scripts that regenerate our dataset let a user reconfigure it — changing resolution, frame rate, the set of emulated sensors, or their parameters — to produce a dataset tailored to their application. Implementation details of the generation interface are given in Appendix S.7.

## 8 Limitations and Outlook

VisionSIM aims to model the raw measurement formation of a broad range of conventional and emerging modalities, together with their characteristic noise and failure modes. The goal is to narrow the gap between idealized synthetic data and real sensor measurements, and support sensor-aware design and analysis that idealized simulators cannot.

Scope of the current results. The digital twin case-study is demonstrated on a single commercial device and a small set of scenes; broader validation across devices, scenes, and operating conditions remains future work. The sensor-suite study is a preliminary, single-task case study (object detection under a CMOS-plus-event versus SPAD-plus-IMU comparison) rather than a comprehensive design analysis. All current evaluations are open-loop: we do not yet close the perception–control loop on a physical or simulated robot. Finally, the released dataset is indoor-only; adding outdoor scenes across various operating regimes is an important next step.

Architectural limitations. Decoupling world simulation from sensor emulation requires an intermediate abstraction (e.g., depth and albedo for dToF). This enables efficient emulation of sensorlevel effects such as pileup, dead time, and jitter, and provides a favorable compromise between speed, modularity, and physical accuracy. However, it precludes direct emulation of volumetric effects and higher-order light transport, as required for modeling complex weather phenomena. Because world simulation is ray-tracing-based, wave-propagation effects such as diffraction are also difficult to capture [62], complicating extension to modalities such as mmWave radar.

Outlook. A future direction is to integrate VisionSIM with real-time engines such as Unreal Engine [4], which could reduce rendering cost and broaden material and asset variety, while moving toward closed-loop and eventually real-time simulation. Incorporating wave-based propagation would open the door to radar and other diffraction-limited sensors, and a library of calibrated sensor presets would let widely-deployed devices be emulated out of the box. We see VisionSIM as a step toward making sensor selection and perception-system design an evidence-based discipline, and we release the platform, assets, and generation scripts toward that goal.

## Acknowledgments

This research was supported in parts by NSF CAREER Award 1943149, NSF grants CNS-2107060, 2107236 and 2513219, the NSF-NIFA AI Institute for Resilient Agriculture, NRT Award Number 2152163, ONR grant N000142412155, and Wisconsin Alumni Research Foundation via a Research Forward Initiative. We would additionally like to thank Varun Sundar and Sungjin Cheong for their help with data collection, and Mani Ramanagopal for insightful discussions on thermal simulation.

## References

[1] Todorov et al. Mujoco: A physics engine for model-based control. In IEEE IROS, 2012.

[2] V. Makoviychuk, L. Wawrzyniak, Y. Guo, M. Lu, K. Storey, M. Macklin, D. Hoeller, N. Rudin, A. Allshire, A. Handa, and G. State. Isaac gym: High performance gpubased physics simulation for robot learning. CoRR, abs/2108.10470, 2021. URL https: //arxiv.org/abs/2108.10470.

[3] G. Authors. Genesis: A universal and generative physics engine for robotics and beyond, December 2024. URL https://github.com/Genesis-Embodied-AI/Genesis.

[4] Epic Games. Unreal engine, 2019. URL https://www.unrealengine.com.

[5] B. O. Community. Blender - a 3d modelling and rendering package, 2018. URL http: //www.blender.org.

[6] A. Gupta, A. Ingle, and M. Gupta. Asynchronous single-photon 3d imaging. IEEE ICCV, 2019.

[7] H. Rebecq, D. Gehrig, and D. Scaramuzza. ESIM: an open event camera simulator. Conf. on Robotics Learning (CoRL), Oct. 2018.

[8] Y. Hu, S. C. Liu, and T. Delbruck. v2e: From video frames to realistic DVS events. In 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2021. URL http://arxiv.org/abs/2006.07722.

[9] A. Upadhyay, M. sharma, P. Mukherjee, A. Singhal, and B. Lall. A comprehensive survey on synthetic infrared image synthesis, 2024. URL https://arxiv.org/abs/2408.06868.

[10] J. R. Schott, S. D. Brown, R. V. Raqueno, H. N. Gross, and G. Robinson. An ad-˜ vanced synthetic image generation model and its application to multi/hyperspectral algorithm development. Canadian Journal of Remote Sensing, 25(2):99–111, 1999. doi: 10.1080/07038992.1999.10874709.

[11] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun. CARLA: An open urban driving simulator. In Proceedings of the 1st Annual Conference on Robot Learning, pages 1–16, 2017.

[12] S. Shah, D. Dey, C. Lovett, and A. Kapoor. Airsim: High-fidelity visual and physical simulation for autonomous vehicles. In International Symposium on Field and Service Robotics, 2017. URL https://api.semanticscholar.org/CorpusID:20999239.

[13] M. Savva, A. Kadian, O. Maksymets, Y. Zhao, E. Wijmans, B. Jain, J. Straub, J. Liu, V. Koltun, J. Malik, D. Parikh, and D. Batra. Habitat: A platform for embodied AI research. CoRR, abs/1904.01201, 2019. URL http://arxiv.org/abs/1904.01201.

[14] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. nuscenes: A multimodal dataset for autonomous driving. arXiv preprint arXiv:1903.11027, 2019.

[15] J. Geyer, Y. Kassahun, M. Mahmudi, X. Ricou, R. Durgesh, A. S. Chung, L. Hauswald, V. H. Pham, M. Muhlegg, S. Dorn, T. Fernandez, M. J¨ anicke, S. Mirashi, C. Savani, M. Sturm,¨ O. Vorobiov, M. Oelker, S. Garreis, and P. Schuberth. A2D2: audi autonomous driving dataset. CoRR, abs/2004.06320, 2020. URL https://arxiv.org/abs/2004.06320.

[16] P. Sun, H. Kretzschmar, X. Dotiwalla, A. Chouard, V. Patnaik, P. Tsui, J. Guo, Y. Zhou, Y. Chai, B. Caine, V. Vasudevan, W. Han, J. Ngiam, H. Zhao, A. Timofeev, S. Ettinger, M. Krivokon, A. Gao, A. Joshi, Y. Zhang, J. Shlens, Z. Chen, and D. Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2020.

[17] A. Geiger, P. Lenz, C. Stiller, and R. Urtasun. Vision meets robotics: The kitti dataset. International Journal ofRobotics Research (IJRR), 2013.

[18] D. J. Butler, J. Wulff, G. B. Stanley, and M. J. Black. A naturalistic open source movie for optical flow evaluation. In A. Fitzgibbon et al. (Eds.), editor, European Conf. on Computer Vision (ECCV), Part IV, LNCS 7577, pages 611–625. Springer-Verlag, Oct. 2012.

[19] L. Mehl, J. Schmalfuss, A. Jahedi, Y. Nalivayko, and A. Bruhn. Spring: A high-resolution high-detail dataset and benchmark for scene flow, optical flow and stereo. In Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023.

[20] D. Scharstein, H. Hirschmuller, Y. Kitajima, G. Krathwohl, N. Ne ¨ siˇ c, X. Wang, and P. West- ´ ling. High-resolution stereo datasets with subpixel-accurate ground truth. In X. Jiang, J. Hornegger, and R. Koch, editors, Pattern Recognition, pages 31–42, Cham, 2014. Springer International Publishing. ISBN 978-3-319-11752-2.

[21] N. Mayer, E. Ilg, P. Hausser, P. Fischer, D. Cremers, A. Dosovitskiy, and T. Brox. A¨ large dataset to train convolutional networks for disparity, optical flow, and scene flow estimation. In IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2016. URL http://lmb.informatik.uni-freiburg.de/Publications/ 2016/MIFDB16. arXiv:1512.02134.

[22] NVIDIA. Nvidia physx, 2020. URL https://developer.nvidia.com/physx-sdk.

[23] K. Greff, F. Belletti, L. Beyer, C. Doersch, Y. Du, D. Duckworth, D. J. Fleet, D. Gnanapragasam, F. Golemo, C. Herrmann, T. Kipf, A. Kundu, D. Lagun, I. Laradji, Hsueh-Ti, Liu, H. Meyer, Y. Miao, D. Nowrouzezahrai, C. Oztireli, E. Pot, N. Radwan, D. Rebain, S. Sabour, M. S. M. Sajjadi, M. Sela, V. Sitzmann, A. Stone, D. Sun, S. Vora, Z. Wang, T. Wu, K. M. Yi, F. Zhong, and A. Tagliasacchi. Kubric: A scalable dataset generator, 2022. URL https://arxiv.org/abs/2203.03570.

[24] E. Coumans and Y. Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. http://pybullet.org, 2016–2021.

[25] M. Denninger, M. Sundermeyer, D. Winkelbauer, Y. Zidan, D. Olefir, M. Elbadrawy, A. Lodhi, and H. Katam. Blenderproc. CoRR, abs/1911.01911, 2019. URL http: //arxiv.org/abs/1911.01911.

[26] C. Bruschini, H. Homulle, I. M. Antolovic, S. Burri, and E. Charbon. Single-photon spad imagers in biophotonics: Review and outlook, 2019.

[27] S. Ma, S. Gupta, A. C. Ulku, C. Brushini, E. Charbon, and M. Gupta. Quanta burst photography. ACM Transactions on Graphics (TOG), 39(4), 7 2020. doi:10.1145/3386569.3392470.

[28] A. Ingle, A. Velten, and M. Gupta. High flux passive imaging with single photon sensors. In Proc. CVPR, June 2019.

[29] M. Ramanagopal, S. Narayanan, A. C. Sankaranarayanan, and S. G. Narasimhan. A theory of joint light and heat transport for lambertian scenes. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11924–11933, 2024. doi:10.1109/ CVPR52733.2024.01133.

[30] K. Tanaka, N. Ikeya, T. Takatani, H. Kubo, T. Funatomi, V. Ravi, A. Kadambi, and Y. Mukaigawa. Time-resolved far infrared light transport decomposition for thermal photometric stereo. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(6): 2075–2085, 2021. doi:10.1109/TPAMI.2019.2959304.

[31] Z. Huang, T. Zhang, W. Heng, B. Shi, and S. Zhou. Real-time intermediate flow estimation for video frame interpolation. In Proceedings of the European Conference on Computer Vision (ECCV), 2022.

[32] M. Tancik, E. Weber, E. Ng, R. Li, B. Yi, J. Kerr, T. Wang, A. Kristoffersen, J. Austin, K. Salahi, A. Ahuja, D. McAllister, and A. Kanazawa. Nerfstudio: A modular framework for neural radiance field development. In ACM SIGGRAPH 2023 Conference Proceedings, SIGGRAPH ’23, 2023.

[33] A. Brussee, F. Hahlbohm, dependabot[bot], V. Fretin, D. N. Hidalgo, E. Michel, B. Ramlot, J. Schmitz, K. Chen, K. Northoff, M. Lanxinger, M. Valigursky, M. Hiltunen, M. Mikovsky, M. van den Berg, P. Diaz, and S. Bethke. Brush: 3d reconstruction for all. https://github.com/ArthurBrussee/brush, 2026. URL https://github.com/ ArthurBrussee/brush.

[34] O. Inc. Digital Lidar Sensors for Automation, Drones & Robotics — Ouster — Ouster — ouster.com. https://ouster.com/.

[35] H. Technology. Global Leader in LiDAR Sensor Solutions — HESAI Technology — hesaitech.com. www.hesaitech.com/.

[36] P. B. Coates. The correction for photon ‘pile-up’ in the measurement of radiative lifetimes. Journal of Physics E: Scientific Instruments, 1(8):878, aug 1968. doi:10.1088/0022-3735/1/ 8/437. URL https://doi.org/10.1088/0022-3735/1/8/437.

[37] D. V. O’Connor and D. Phillips. Time-Correlated Single Photon Counting. Elsevier, 1984. ISBN 9780125241403. doi:10.1016/b978-0-12-524140-3.x5001-1. URL http://dx.doi. org/10.1016/b978-0-12-524140-3.x5001-1.

[38] S. Inc. Time of Flight Image Sensor — Products & Solutions — Sony Semiconductor Solutions Group — sony-semicon.com. https://www.sony-semicon.com/en/products/ is/industry/tof.html.

[39] STMicroelectronics. STMicroelectronics Time-of-Flight sensors. https://www.st.com/ en/imaging-and-photonics-solutions/time-of-flight-sensors.html.

[40] R. Lange and P. Seitz. Solid-state time-of-flight range camera. IEEE Journal of Quantum Electronics, 37(3):390–397, 2001. doi:10.1109/3.910448.

[41] M. Gupta, A. Velten, S. K. Nayar, and E. Breitbach. What are optimal coding functions for time-of-flight imaging? ACM Trans. Graph., 37(2):13:1–13:18, Feb. 2018. ISSN 0730-0301. doi:10.1145/3152155. URL http://doi.acm.org/10.1145/3152155.

[42] R. Sawhney and B. Miller. Zombie: Grid-free monte carlo solvers for partial differential equations, 2023.

[43] ADAPS Photonics. Ads6311 large spad array dtof chip and module. https://www. adapsphotonics.com/en/product-55669-218658.html, 2023. Accessed: 2026-05-13.

[44] H. Sim, J. Oh, and M. Kim. Xvfi: extreme video frame interpolation. In Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2021.

[45] Y. Wang, Z. Xian, F. Chen, T.-H. Wang, Y. Wang, K. Fragkiadaki, Z. Erickson, D. Held, and C. Gan. Robogen: Towards unleashing infinite data for automated robot learning via generative simulation, 2023.

[46] Z. Sun, X. Fu, L. Huang, A. Liu, and Z.-J. Zha. Motion aware event representation-driven image deblurring. In Proceedings of the European conference on computer vision (ECCV). Springer, 2024.

[47] I. Robinson, P. Robicheaux, M. Popov, D. Ramanan, and N. Peri. Rf-detr: Neural architecture search for real-time detection transformers, 2025. URL https://arxiv.org/abs/2511. 09554.

[48] N. A. W. Dutton, I. Gyongy, L. Parmesan, and R. K. Henderson. Single photon counting performance and noise analysis of cmos spad-based image sensors. Sensors, 16(7), 2016. ISSN 1424-8220. doi:10.3390/s16071122. URL https://www.mdpi.com/1424-8220/ 16/7/1122.

[49] Y. Nozaki and T. Delbruck. Temperature and parasitic photocurrent effects in dynamic vision sensors. IEEE Transactions on Electron Devices, 64(8):3239–3245, 2017. doi:10.1109/TED. 2017.2717848.

[50] P. Lichtsteiner, C. Posch, and T. Delbruck. A 128× 128 120 db 15 µs latency asynchronous temporal contrast vision sensor. IEEE Journal of Solid-State Circuits, 43(2):566–576, 2008. doi:10.1109/JSSC.2007.914337.

[51] M. Grossberg and S. Nayar. Modeling the space of camera response functions. IEEE Transactions on Pattern Analysis and Machine Intelligence, 26(10):1272–1282, 2004. doi: 10.1109/TPAMI.2004.88.

[52] I. Brouk, A. Nemirovsky, and Y. Nemirovsky. Analysis of noise in cmos image sensor. In 2008 IEEE International Conference on Microwaves, Communications, Antennas and Electronic Systems, pages 1–8, 2008. doi:10.1109/COMCAS.2008.4562800.

[53] S. Leutenegger, S. Lynen, M. Bosse, R. Siegwart, and P. Furgale. Keyframe-based visual–inertial odometry using nonlinear optimization. Int. J. Rob. Res., 34(3):314–334, Mar. 2015. ISSN 0278-3649. doi:10.1177/0278364914554813. URL https://doi.org/10. 1177/0278364914554813.

[54] S. Hwang, J. Park, N. Kim, Y. Choi, and I. S. Kweon. Multispectral pedestrian detection: Benchmark dataset and baseline. In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1037–1045, 2015. doi:10.1109/CVPR.2015.7298706.

[55] D. Rodr´ıguez-Mart´ınez, D. van der Meer, J. Song, A. Bera, C. J. Perez-del Pulgar, and M. A.´ Olivares-Mendez. Spice-hl3: Single-photon, inertial, and stereo camera dataset for exploration of high-latitude lunar landscapes. Scientific Data, 13(1):374, 2026. doi:10.1038/ s41597-026-06668-8. URL https://doi.org/10.1038/s41597-026-06668-8.

[56] Y. Verdie, J. Song, B. Mas, B. Busam, A. Leonardis, and S. McDonagh. Cromo: Cross-modal´ learning for monocular depth estimation, 2022.

[57] Y. Han, Z. Liu, S. Sun, D. Li, J. Sun, Z. Hong, and M. H. Ang Jr. Carla-loc: Synthetic slam dataset with full-stack sensor setup in challenging weather and dynamic environments. arXiv preprint arXiv:2309.08909, 2023.

[58] W. Wang, D. Zhu, X. Wang, Y. Hu, Y. Qiu, C. Wang, Y. Hu, A. Kapoor, and S. Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), page 4909–4916. IEEE, Oct. 2020. doi:10.1109/iros45743.2020.9341801. URL http://dx.doi.org/10.1109/iros45743. 2020.9341801.

[59] T. Klinghoffer, S. Somasundaram, X. Xiang, Y. Fan, C. Richardt, A. Dave, R. Raskar, and R. Ranjan. Shoot-bounce-3d: Single-shot occlusion-aware 3d from lidar by decomposing two-bounce light. In Proceedings of the SIGGRAPH Asia 2025 Conference Papers, SA Conference Papers ’25, New York, NY, USA, 2025. Association for Computing Machinery. ISBN 9798400721373. doi:10.1145/3757377.3763945. URL https://doi.org/10. 1145/3757377.3763945.

[60] M. A. Farooq, W. Shariff, and P. Corcoran. Thermvision-db: A synthetic lwir thermal face dataset for privacy-preserving thermal vision research. Data in Brief, 65:112506, 2026. ISSN 2352-3409. doi:https://doi.org/10.1016/j.dib.2026.112506. URL https: //www.sciencedirect.com/science/article/pii/S2352340926000594.

[61] Q. Guo, I. Frosio, O. Gallo, T. Zickler, and J. Kautz. Tackling 3d tof artifacts through learning and the flat dataset. In Computer Vision – ECCV 2018: 15th European Conference, Munich, Germany, September 8-14, 2018, Proceedings, Part I, page 381–396, Berlin, Heidelberg, 2018. Springer-Verlag. ISBN 978-3-030-01245-8. doi:10.1007/978-3-030-01246-5 23. URL https://doi.org/10.1007/978-3-030-01246-5\_23.

[62] J. B. Keller. Geometrical theory of diffraction∗. J. Opt. Soc. Am., 52(2):116–130, Feb 1962. doi:10.1364/JOSA.52.000116. URL https://opg.optica.org/abstract.cfm? URI=josa-52-2-116.

[63] Y. Liu, F. Gutierrez-Barragan, A. Ingle, M. Gupta, and A. Velten. Single-photon camera guided extreme dynamic range imaging. In Proceedings of the IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 1575–1585, January 2022.

[64] W. Jakob, S. Speierer, N. Roussel, M. Nimier-David, D. Vicini, T. Zeltner, B. Nicolet, M. Crespo, V. Leroy, and Z. Zhang. Mitsuba 3 renderer, 2022. https://mitsuba-renderer.org.

[65] Center for High Throughput Computing. Center for high throughput computing, 2006. URL https://chtc.cs.wisc.edu/.

[66] D. Merkel. Docker: lightweight linux containers for consistent development and deployment. Linux J., 2014(239), Mar. 2014. ISSN 1075-3583.

[67] A. Dosovitskiy, P. Fischer, E. Ilg, P. Hausser, C. Hazırbas¸, V. Golkov, P. v.d. Smagt, D. Cre-¨ mers, and T. Brox. Flownet: Learning optical flow with convolutional networks. In IEEE International Conference on Computer Vision (ICCV), 2015. URL http://lmb.informatik. uni-freiburg.de/Publications/2015/DFIB15.

[68] S. Narayanan, M. Ramanagopal, M. Sheinin, A. C. Sankaranarayanan, and S. G. Narasimhan. Shape from heat conduction. In Computer Vision – ECCV 2024, pages 426–444. Springer Nature Switzerland, 2025. doi:10.1007/978-3-031-72920-1 24.

[69] ANSYS, Inc. ANSYS Academic Research Mechanical, Release 2024 R2. ANSYS, Inc., Canonsburg, PA, USA, 2024. https://www.ansys.com.

[70] COMSOL AB. COMSOL Multiphysics®, Version 6.2. COMSOL AB, Stockholm, Sweden, 2024. https://www.comsol.com.

[71] OpenFOAM Foundation. OpenFOAM: The Open Source CFD Toolbox. OpenCFD Ltd., 2024. https://www.openfoam.com.

[72] N. Sharp and K. Crane. A Laplacian for Nonmanifold Triangle Meshes. Computer Graphics Forum (SGP), 39(5), 2020.

[73] R. Bridson. Fast poisson disk sampling in arbitrary dimensions. In ACM SIGGRAPH 2007 Sketches, SIGGRAPH ’07, page 22–es, New York, NY, USA, 2007. Association for Computing Machinery. ISBN 9781450347266. doi:10.1145/1278780.1278807. URL https://doi.org/10.1145/1278780.1278807.

[74] M. Wardetzky, S. Mathur, F. Kalberer, and E. Grinspun. Discrete laplace operators: no free¨ lunch. In Proceedings of the Fifth Eurographics Symposium on Geometry Processing, SGP ’07, page 33–37, Goslar, DEU, 2007. Eurographics Association. ISBN 9783905673463.

[75] Blender demo gallery, 2024. URL https://www.blender.org/download/demo-files/.

[76] F. Reda, J. Kontkanen, E. Tabellion, D. Sun, C. Pantofaru, and B. Curless. Film: Frame interpolation for large motion. In European Conference on Computer Vision (ECCV), 2022.

[77] A. Telea. An image inpainting technique based on the fast marching method. Journal of Graphics Tools, 9, 01 2004. doi:10.1080/10867651.2004.10487596.

[78] A. Levin, D. Lischinski, and Y. Weiss. Colorization using optimization. ACM Trans. Graph., 23(3):689–694, Aug. 2004. ISSN 0730-0301. doi:10.1145/1015706.1015780. URL https: //doi.org/10.1145/1015706.1015780.

[79] A. Carlson, K. A. Skinner, R. Vasudevan, and M. Johnson-Roberson. Sensor transfer: Learning optimal sensor effect image augmentation for sim-to-real domain adaptation. CoRR, abs/1809.06256, 2018. URL http://arxiv.org/abs/1809.06256.

[80] S. Park, J. Pahk, L. L. F. Jahn, Y. Lim, J. An, and G. Choi. A study on quantifying sim2real image gap in autonomous driving simulations using lane segmentation attention map similarity, 2023. URL https://arxiv.org/abs/2306.10491.

[81] S. W. Hasinoff, F. Durand, and W. T. Freeman. Noise-optimal capture for high dynamic range photography. In 2010 IEEE Computer Society Conference on Computer Vision and Pattern Recognition, pages 553–560, San Francisco, CA, USA, June 2010. IEEE. ISBN 978- 1-4244-6984-0. doi:10.1109/CVPR.2010.5540167. URL http://ieeexplore.ieee.org/ document/5540167/.

[82] S. Jungerman, V. Sundar, and M. Gupta. A live demo of single-photon imaging and applications. In ACM SIGGRAPH 2024 Emerging Technologies, SIGGRAPH ’24, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400705243. doi: 10.1145/3641517.3664397. URL https://doi.org/10.1145/3641517.3664397.

[83] A. C. Ulku, C. Bruschini, I. M. Antolovic, E. Charbon, Y. Kuo, R. Ankri, S. Weiss, and X. Michalet. A 512×512 spad image sensor with integrated gating for widefield flim. IEEE journal of selected topics in quantum electronics : a publication of the IEEE Lasers and Electro-optics Society, 25(1):6801212, 2019. ISSN 1077-260X. doi:10.1109/jstqe.2018. 2867439. URL https://europepmc.org/articles/PMC6541425.

[84] A. Ingle and D. Maier. Count-free single-photon 3D imaging with race logic. IEEE Transactions on Pattern Analysis and Machine Intelligence, pages 1–12, 2023. doi:10.1109/TPAMI. 2023.3302822.

[85] K. Sadekar, D. Maier, and A. Ingle. Single-Photon 3D Imaging with Equi-Depth Photon Histograms. In European Conference on Computer Vision, pages 381–398. Springer, 2024.

[86] A. Glover and C. Bartolozzi. Robust visual tracking with a freely-moving event camera. In 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 3769–3776, 2017. doi:10.1109/IROS.2017.8206226.

[87] N. Messikommer\*, C. Fang\*, M. Gehrig, and D. Scaramuzza. Data-driven feature tracking for event cameras. IEEE Conference on Computer Vision and Pattern Recognition, 2023.

[88] J. Crassidis. Sigma-point kalman filtering for integrated gps and inertial navigation. IEEE Transactions on Aerospace and Electronic Systems, 42(2):750–756, 2006. doi:10.1109/ TAES.2006.1642588.

[89] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza. Imu preintegration on manifold for efficient visual-inertial maximum-a-posteriori estimation. In Robotics: Science and Systems XI, 2015.

[90] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.

[91] J. L. Schonberger and J.-M. Frahm. Structure-from-motion revisited. In¨ Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[92] S. Jungerman, A. Garg, and M. Gupta. Radiance fields from photons, 2024. URL https: //arxiv.org/abs/2407.09386.

[93] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis. 3d gaussian splatting for real-time¨ radiance field rendering. ACM Transactions on Graphics, 42(4), July 2023. URL https: //repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.

[94] T. Muller, A. Evans, C. Schied, and A. Keller. Instant neural graphics primitives with a ¨ multiresolution hash encoding. ACM Trans. Graph., 41(4):102:1–102:15, July 2022. doi: 10.1145/3528223.3530127. URL https://doi.org/10.1145/3528223.3530127.

[95] S. Narayanan, M. Ramanagopal, and S. Narasimhan. Dual band thermal videography: Separating time-varying reflection and emission near ambient conditions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 199–208, June 2026.

[96] F. Sakuma. Establishing a practical temperature standard by using a narrow-bannd radiation thermometer with a silicon detector. Temperature, Its Measurement and Control in Science and Industry, 5:421–427, 1982.

[97] M. Srinivasan Ramanagopal, Z. Zhang, R. Vasudevan, and M. Johnson Roberson. Pixelwise motion deblurring of thermal videos. In Robotics: Science and Systems XVI, RSS2020. Robotics: Science and Systems Foundation, July 2020. doi:10.15607/rss.2020.xvi.022. URL http://dx.doi.org/10.15607/rss.2020.xvi.022.

[98] M. Vollmer. Infrared thermal imaging. In Computer Vision: A Reference Guide, pages 1–4. Springer, 2020.

[99] N. Topaloglu. Analysis and modeling of uncooled microbolometers with tunable thermal conductance. 2009.

[100] F. Niklaus, A. Decharat, C. Jansson, and G. Stemme. Performance model for uncooled infrared bolometer arrays and performance predictions of bolometers operating at atmospheric pressure. Infrared Physics & Technology, 51(3):168–177, 2008. ISSN 1350-4495. doi: https://doi.org/10.1016/j.infrared.2007.08.001. URL https://www.sciencedirect.com/ science/article/pii/S1350449507000795.

[101] M. Muldoon and J. Roper. Blendswap, 20131. URL https://blendswap.com/.

# VisionSIM: Beyond Idealized Sensors — A Multi-Modal Simulator for Robot Perception Supplementary Material

## S.1 Architecture

VisionSIM is designed from the ground up to be maximally flexible. A key design principle is its multi-layered architecture that enables it to simulate a variety of seemingly unrelated sensor modalities (including unconventional and emerging sensors such as single-photon cameras which excel in high-speed and high dynamic range environments [27, 63]) with minimal effort by reusing its different components and extending them with plug-ins if needed.

The simulation engine is made up of layers that are accessible both (a) as a command-line interface tool, which is useful to complete common operations such as rendering different types of ground truth data or emulating conventional cameras, and (b) as a full-fledged library, which enables more complicated workflows such as simulating multi-camera rigs. These layers are as follows:

World Simulation Layer: Generate rich ground truth data annotations using high-quality rendering engines, optionally interpolate to higher frame rates to emulate realistic motion blur and high-speed sensors.

Sensor Emulation Layer: Emulate different sensor modalities using realistic image-formation models, enabling physically accurate sensor noise.

Once simulated, new datasets are stored with all relevant metadata in a way that enables easy iteration and random access, enabling easy interoperability with existing frameworks.

While the default workflow is to use these layers in sequence to manipulate the data until it is in the desired format, these can be used individually and even out of order to support different applications. For instance, if we skip the simulation layer and instead provide real high-speed video captures (e.g. from XVFI [44]), we can emulate high-speed sensing modalities (e.g. event cameras and singlephoton cameras), or conventional cameras from this dataset using the sensor emulation layer.

## S.2 World simulation

The first step is to render out clean, blur and noise-free RGB images, depth maps, as well as normal maps, and other data modalities where applicable, which will both enable the emulation of different sensor types as well as act as ground truth.

While there are many high-quality rendering engines such as Unreal Engine [4] or Mitsuba3 [64], we opt to use Blender [5] as our primary rendering engine as it is easy to integrate with, widely accessible, well supported and produces photorealistic images.

However, Blender’s built-in API is very complex and not tailored towards high frame-rate rendering. It has a built-in maximum frame rate – which is not sufficient to model event cameras or singlephoton cameras natively, the latter of which can operate at hundreds of thousands of frames per second – and has no easy way to output some ground truth annotations such as camera intrinsics.

Further, rendering can be very costly and this cost scales with frame rate. By default, when a frame is rendered system utilization can be near a hundred percent, yet in between frames, or when setting up the scene for the next render, utilization is nearly zero, leading to spiky system load and underutilized resources further compounding slow render times. Finally, authoring new assets that target high fps is difficult as graphic designers are not used to doing so, and most existing assets would either playback at unrealistic speeds or be completely broken if they relied on more complicated effects such as geometry nodes or custom shaders.

![](images/208a2a9563ab5460957985cdb6f86faae2b05f39d0844f73be71f58fd07c5ccb.jpg)

(b) Interpolation Artifacts at Various Frame Rates  
![](images/7979632216a5ca51f84dd7e2f0b063509ce5750d95172995185e7d2fdbe007d1.jpg)  
(c) Application Programming Interface and Code Examples

![](images/66292e1caaff48f4e55fc6b5b882b83bd49320d67100fb4f4b8fb23579ffb15c.jpg)  
Suppl. Fig. 1: Scalable Rendering and Interpolation: (A) Client/server rendering architecture: multiple rendering services can be spun up on local or remote machines, each exposing a highlevel rendering API via remote procedure calls (RPCs) and broadcasting their addresses over UDP for automated discovery via a registry server. This enables multiple clients to be orchestrated in tandem to distribute rendering jobs and speed up processing. (B) Interpolation quality/speed tradeoff: we render a scene at different frame rates and interpolate using RIFE [31] or simple frame duplication until we reach 6400 fps. A low rendering rate can introduce severe artifacts, shown here quantitatively (left) as a high perceptual similarity score (lower is better) and qualitatively (right), while a fast rate wastes compute for marginal quality gains. (C) Create a pool of two render workers and distribute work amongst them. For each of the three blender files, we enable the rendering of normals (L5), slow down the animations by a factor of ten (L6), and then render everything. By instead freeing the camera from any constraints we can move it manually and generate any arbitrary view. Here we spawn and connect to a single render client and fly the camera in a circle around the origin while always looking towards the origin. We also change the resolution (L3), render out depth buffers (L4), and disable any animations (L5).

To address these challenges, we introduce a streamlined rendering API that simplifies the dataset generation workflow, makes it easy to reuse or create assets that have been authored for slower conventional frame rates, and allows for fast, distributed rendering on heterogeneous hardware.

## S.2.1 Streamlined API

Blender provides an extensive python API, namely bpy, that exposes virtually all of Blender’s functionality. However, it is only accessible through Blender’s internal Python distribution or as a Python module if the user compiles Blender from its source. We instead provide a simplified API which provides a higher-level interface that streamlines the dataset generation process and automatically handles common pitfalls users might encounter.

This simplified API makes no assumptions about the scene that is loaded and will by default attempt to use all render settings the user set within Blender. While a new rendering pass, along with all necessary compositing nodes, might be automatically added to support different ground truth annotations, it is the user’s responsibility to provide functional blend files. In Fig. 1 (C, left), multiple animations are rendered, and a rendering pass to get surface normals is automatically added if needed. However, an animation will not be rendered if its corresponding blender file is malformed in such a way that manually rendering a frame from within Blender does not work. Further, this simplified API makes it easy for users to generate different types of ground truth annotations and change common rendering settings such as targeting a different render device or changing resolution, as well as more advanced operations like manual camera updates within the render loop.

## S.2.2 Distributed rendering

To enable scalable simulations, we employ a client/server architecture, shown in Fig. 1 (A), that uses remote procedure calls (RPCs) as the communication medium. A few minimal dependencies are injected within Blender’s runtime, which enables it to run as a server, listening for requests to set up scenes, move around cameras, render data, etc. Specifically, doing so from within Blender’s runtime enables us to access all features through the bpy package and expose them via RPCs.

Correctly orchestrating many rendering servers is technically challenging. First, to enable easy discovery of servers, a registry server is spawned and listens for UDP-based broadcast messages from rendered servers that are available. This rendezvous server communicates with the main client to inform it about available workers, enabling the user to connect to the server instances and distribute work amongst the worker pool, or even use multiple render servers to render out a single animation. Servers manage their own logging and exception handling, and, in the event of some unexpected crash, we ensure that the dataset is not corrupted and the process resources are freed.

This client/server architecture can drastically cut down on processing time and enables the use of a distributed set of machines with heterogeneous hardware. Even when worker processes are on the same machine, as is the case when using BlenderClients’ spawn or pool directives (Fig. 1 (C)), render times can be shorter as the system’s resources will be better utilized. Such distributed capability is a novel contribution to the Blender ecosystem as it provides the tooling for scalable and distributed renders – which many companies, render farms, and add-ons provide as a paid service – as a free and open-source library, which will democratize scaling of simulation-based bootstrapping of ML models.

For even larger-scale dataset generation tasks such as the dataset in Section 7, we employ a second layer of distributed computing through the use of HTCondor [65], a high-throughput computing framework widely deployed in academic and institutional clusters. This involves containerizing our entire simulation stack, including all scripts, dependencies, and assets, using Docker [66], which is automated in CI/CD, and submitting jobs to an HTCondor access point. This setup allowed us to leverage thousands of CPU cores and hundreds of GPUs across thousands of machines to render out the VisionSIM dataset described in Sec. 7.

## S.2.3 Keyframe remapping

To address the challenges related to authoring high-speed assets and overcome Blender’s maximum frame rate limitations, we propose to dynamically remap keyframes. In effect, keyframes are the control points that define the splines that govern all animations and dynamics, and, by remapping them, we can dilate time as needed. This enables easy re-use of existing virtual scenes and ensures that new frames do not break any pre-existing animations, as they are simply interpolated between keyframes. An example of this can be seen in Fig. 1 (C, left).

## S.2.4 Physics engine

Importantly, the keyframe-remapping technique enables us to take full advantage of Blender’s powerful physics simulation engine. While it is slower than others [2, 3], the Bullet [24] physics engine is extremely powerful, especially since it can be customized using Blender’s flexible geometry nodes, enabling users to build complex physical systems procedurally. Effects like fluid and cloth simulations, soft or rigid body physics, and more can simply be baked as keyframes and then remapped to any arbitrary frame rate. This enables us to easily create Flying Chairs [67]-like datasets at much higher frame rates without needing to explicitly tweak the physics sub-time-stepping or collision engine. This is also the same physics engine used by Kubric [23], although they manually integrate it into their pipeline using Bullet’s Python API, restricting simulations to only rigid body physics.

## S.2.5 Heat Transport Simulation

Several sensor modalities of interest—most notably long-wave infrared (LWIR, 8–14 µm) thermal cameras—respond not to the instantaneous radiance leaving a surface, but to the long-wave radiation it emits as a consequence of its temperature, and that temperature is the accumulated result of every watt of power the surface has absorbed over the preceding seconds to hours. Synthesizing a thermal image therefore requires synthesizing a plausible temperaturefield first, which the sensor emulation layer (Sec. S.5.4) then renders into a measurement. To support these modalities, VisionSIM integrates a fast, GPU-accelerated heat-transport solver directly into the world simulation layer, reusing the same Blender asset-authoring and rendering pipeline described above so that the temperature field is always consistent with the exact lighting the scene was authored with. This section develops the forward model in two stages: how absorbed visible light becomes a heat source (Sec. S.2.5.1), and how the resulting heat-transport PDE is discretized and solved (Sec. S.2.5.2).

## S.2.5.1 Visible Light to Thermal Heat Source

The bridge between the visible-band light transport that VisionSIM already simulates and the thermal domain is energy conservation at an opaque surface. The pixel value an ordinary camera records is the fraction of incident light reflected toward it; since an opaque surface transmits nothing, the remaining energy is absorbed, converted to heat, and subsequently redistributed by conduction, convection, and re-radiation, governed by the heat-transport equation [68]. Reflection and absorption are thus complementary [29], and the same path-traced irradiance the renderer evaluates to form the visible image also determines the heat deposited at each surface. Making this precise for a Lambertian scene point x, let $E ( \mathbf { x } , t )$ denote the incident visible-band radiant flux density $( \mathrm { W / m ^ { 2 } } )$ arriving at x under the scene’s lighting. A camera focused at x measures only the reflected portion, giving the visible image intensity

$$
I _ { v } ( \mathbf { p } ) = \frac { a ( \mathbf { x } ) } { \pi } E ( \mathbf { x } , t ) ,\tag{S1}
$$

where p is the pixel imaging x and $a ( \mathbf { x } ) \in [ 0 , 1 ]$ is the surface albedo. The complementary absorbed power per unit area—the quantity that drives the heat equation—is the remaining fraction,

$$
q _ { \mathrm { a b s } } ( \mathbf { x } , t ) \ = \ \beta ( \mathbf { x } ) E ( \mathbf { x } , t ) , \qquad \beta ( \mathbf { x } ) = 1 - a ( \mathbf { x } ) ,\tag{S2}
$$

which enters the thermal problem as a Neumann boundary source (Sec. S.2.5.2). The broadband absorptivity $\beta ( \mathbf { x } ) \in [ 0 , 1 ]$ generalizes the ideal Lambertian factor 1 − a to account for surface finish and angular reflectance, and reduces to it when the user ties absorption directly to the visible albedo.

The decisive modeling choice is that we obtain E directly from the renderer rather than asking the user to specify a heat flux by hand, as classical FEM packages (ANSYS [69], COMSOL [70], OpenFOAM [71]) require. Concretely, we read E from the renderer’s diffuse irradiance passes— the direct and indirect diffuse light transport that already produces the matte component of the RGB image, including indirect bounces, environment lighting, sky and sun, and animated emitters—baked into a per-texel irradiance map at each timestep and resampled to mesh vertices by barycentric interpolation. The practical consequence is that the temperature field is always consistent with the exact lighting the scene was authored with: moving the sun to a different time of day moves the heat input, adding an emitter adds a heat source, and turning off a lamp removes its thermal contribution—all without the user authoring any thermal boundary condition.

## S.2.5.2 Finite Element Heat Transport Simulation

Governing PDE. We model heat transport in a solid domain $\Omega \subset \mathbb { R } ^ { 3 }$ (following [68]) with boundary ∂Ω, exchanging energy with an ambient at temperature $T _ { \mathrm { a m b } }$ through convection and radiation while absorbing externally driven irradiance on ∂Ω. The temperature field $T ( \mathbf { x } , t )$ satisfies the inhomogeneous heat equation

$$
\rho { c _ { p } } \frac { { \partial T } } { { \partial t } } = \nabla \cdot \left( k \nabla T \right) + \dot { q } , \qquad \mathbf x \in \Omega ,\tag{S3}
$$

where $\rho ( \mathbf { x } ) , c _ { p } ( \mathbf { x } )$ , and $k ( \mathbf { x } )$ are the spatially varying density, specific heat capacity, and thermal conductivity, and q˙ is a volumetric source. We take ${ \dot { q } } \equiv 0$ and inject all energy through the boundary. The transient response is governed by the thermal diffusivity $\alpha = { k } / ( \rho c _ { p } )$ , which is the natural parameter to reason about: aluminum $( \alpha \approx 9 7 \mathrm { { m m } ^ { 2 } / s ) }$ diffuses heat roughly 500× faster than glass $( \alpha \approx 0 . 5 \mathrm { m m ^ { 2 } / s ) }$ , and it is this ratio rather than the conductivity alone that distinguishes their thermal appearance over the timescale of a video.

Boundary conditions. A surface has no thickness and hence zero mass, so it cannot store energy and any heat entering it must exit on the other side. On ∂Ω this balance is the Robin-type condition

$$
\begin{array} { r } { - k \nabla T \cdot \hat { \mathbf { n } } = \underbrace { \sigma \epsilon ( T ^ { 4 } - T _ { \mathrm { a m b } } ^ { 4 } ) } _ { \mathrm { r a d i a t i o n } } + \underbrace { h _ { c } ( T - T _ { \mathrm { a m b } } ) } _ { \mathrm { c o n v e c t i o n } } , \qquad \mathbf { x } \in \partial \Omega , } \end{array}\tag{S4}
$$

where nˆ is the outward unit normal, σ is the Stefan–Boltzmann constant, $\epsilon \in [ 0 , 1 ]$ is the LWIR emissivity, and $h _ { c }$ is the convective heat-transfer coefficient. The left-hand side is the conductive flux through the surface given by Fourier’s law; the right-hand side is the flux lost to radiation and convection. Surfaces with externally imposed temperatures (a poured hot liquid, a thermostatic heater, a contact patch) are instead handled with Dirichlet conditions $T ( \mathbf { x } , t ) = T _ { D } ( \mathbf { x } , t )$ on usertagged sub-regions $\Gamma _ { D } \subset \partial \Omega$ , and the field is initialized from a per-vertex attribute $T ( \mathbf { x } , 0 ) = T _ { 0 } ( \mathbf { x } )$ that defaults to $T _ { \mathrm { a m b } }$

Combined heat equation. Following [68], we fold the boundary condition directly into the energy balance of a finite element exposed to the surface with area A. Adding the absorbed irradiance as an external input gives a single combined equation,

$$
\rho c _ { p } ~ { \frac { \mathrm { d } T } { \mathrm { d } t } } ~ \mathrm { d v } ~ = ~ k \Delta T ~ \mathrm { d v } + \mathbb { 1 } _ { \partial \Omega } { \Big ( } \sigma \epsilon A \left( T _ { \mathrm { a m b } } ^ { 4 } - T ^ { 4 } \right) + h _ { c } A \left( T _ { \mathrm { a m b } } - T \right) + q _ { \mathrm { a b s } } { \Big ) }\tag{S5}
$$

where $\Delta$ is the Laplace–Beltrami operator, $q _ { \mathrm { a b s } }$ is the input power per unit area $\mathrm { ( W / m ^ { 2 } ) }$ absorbed by the object on the boundary in the visible spectrum and $\mathbb { 1 } _ { \partial \Omega }$ is an indicator function which is 1 if the infinitesimal volume is on the surface and 0 otherwise.

Linearized radiation. The quartic in Eq. S5 makes the boundary term nonlinear in T. For smaller temperature differences the radiation term can be linearized without sufficient loss of accuracy [68],

$$
\sigma \epsilon ( T _ { \mathrm { a m b } } ^ { 4 } - T ^ { 4 } ) \approx 4 \sigma \epsilon T _ { \mathrm { a m b } } ^ { 3 } ( T _ { \mathrm { a m b } } - T ) .\tag{S6}
$$

Crucially, this folds radiation into a constant boundary stiffness term that references only the fixed $T _ { \mathrm { a m b } }$ rather than the evolving solution, so the system matrix can be assembled once and reused across the animation, with only the lighting-dependent right-hand side reassembled per frame. For high-temperature targets (welding, furnace interiors etc.) the radiation term can instead be relinearized about the current solution each step, at the cost of per-frame matrix reassembly.

Spatial discretization. We represent the temperature as a piecewise-linear field over the discrete domain, $\begin{array} { r } { T ( \mathbf { x } , t ) \approx \sum _ { i } \varphi _ { i } ( \mathbf { x } ) T _ { i } ( t ) } \end{array}$ , with $\varphi _ { i }$ the nodal basis functions, and discretize Ω into finite elements with V vertices. The discrete Laplace–Beltrami operator is $\begin{array} { r } { \Delta _ { \Omega } = M _ { v } ^ { - 1 } L } \end{array}$ , where $L \succeq 0$ is the cotangent Laplacian and $M _ { v }$ the lumped vertex mass matrix; a second, area-weighted mass matrix $M _ { a }$ , assembled by summing one third of each boundary triangle’s area into its vertices, integrates the boundary terms. Generic asset meshes are typically open, non-manifold, and full of degenerate triangles, which break a naive cotangent Laplacian. We therefore assemble L with an intrinsically mollified cotangent operator [72], which regularizes needle triangles, slivers, and nearcoincident vertices so the system stays well-conditioned on geometry that no conventional FEM package would accept without remeshing.

Time integration. Substituting the linearized radiation Eq. S6 into the discretized combined equation Eq. S5 and applying implicit backward Euler, $\dot { \mathbf { T } } \approx ( \bar { T } ^ { n + 1 } - T ^ { n } ) / t$ , each timestep is a single sparse linear solve. Gathering $T ^ { n + 1 }$ on the left and multiplying through by $M _ { v }$ gives

$$
\begin{array} { r } { \Big ( M _ { v } + \alpha t L + \mathbb { 1 } _ { \partial \Omega } \frac { t } { \rho c _ { p } } \big ( 4 \sigma \epsilon T _ { \mathrm { a m b } } ^ { 3 } + h _ { c } \big ) M _ { a } \Big ) T ^ { n + 1 } \ = \ M _ { v } T ^ { n } } \\ { + \mathbb { 1 } _ { \partial \Omega } \frac { t } { \rho c _ { p } } \big ( 4 \sigma \epsilon T _ { \mathrm { a m b } } ^ { 4 } + h _ { c } T _ { \mathrm { a m b } } + q _ { \mathrm { a b s } } ^ { n + 1 } \big ) M _ { a } \mathbf { 1 } , } \end{array}\tag{S7}
$$

where $\mathbb { 1 } _ { \partial \Omega }$ is the boundary-vertex indicator and the per-vertex scalars $\epsilon , \rho , c _ { p } , \beta , q _ { \mathrm { a b s } } ^ { n + 1 }$ are read from the gathered material vectors. The left-hand matrix is sparse, symmetric positive definite, and— because radiation is linearized about the fixed $T _ { \mathrm { a m b } }$ —constant across the animation; only the righthand side, through $q _ { \mathrm { a b s } } ^ { n + 1 }$ , changes when the lighting is animated. It is therefore prefactored once and the per-step solve is dramatically cheaper, warm-started from the previous timestep with a matrixfree conjugate-gradient iteration that keeps the iteration count low for smoothly evolving lighting.

Volumetric heat simulation. A surface-only solve suffices for thin shells (sheet metal, fabric, leaves), but it misrepresents the diffusion pattern for thicker objects, and it fails outright for the hollow meshes common in asset stores: a mug or a skillet modeled as a thin shell conducts heat as if it were a hollow surface rather than a solid body, producing visibly wrong behavior. One way to fix is by tetrahedralizing the interior, but watertight tetrahedralization is unworkable on the open, selfintersecting meshes these assets typically are. We instead work with points rather than tetrahedra. For each closed (or near-closed) object we sample a dense, blue-noise interior point cloud over the bounding box by Bridson Poisson-disk sampling [73], keeping only points that pass a BVH rayparity inside test: a point is interior iff a random ray crosses the surface an odd number of times, and we take a 256-ray majority vote for robustness to degenerate geometry. A mollified point-cloud Laplacian [72], built on a k-nearest-neighbour graph over the combined surface-plus-interior point set, then couples interior diffusion to the boundary degrees of freedom exactly as a volume FEM would, but with no meshing required. In Eq. S7 the indicator $\mathbb { 1 } _ { \partial \Omega }$ simply selects the original surface vertices: interior points diffuse but receive no irradiance, radiation, or convection. The resulting sparse systems are assembled and solved on the GPU, keeping the solve fast enough to run in the loop of a dataset-generation pipeline; per-vertex temperatures are written back as a standard Blender attribute that the downstream sensor emulation layer (Sec. S.5.4) consumes.

Scope of the heat-transport model. Two modeling choices bound the fidelity of the temperature field. First, we read the absorbed power from the renderer’s diffuse irradiance passes (Eq. S2), neglecting the specular component of the incident radiance; since specularly reflected energy is reflected rather than absorbed, attributing it to $q _ { \mathrm { a b s } }$ causes our heat source to systematically overestimate the power deposited at glossy surfaces. Second, the mollified Laplacian [72] built over a k-nearest-neighbour point set is robust on the open, non-manifold meshes typical of asset stores, but this robustness comes at a cost: no discrete Laplacian can satisfy symmetry, locality, linear precision, and positivity simultaneously [74], and the point-cloud solve forgoes the conforming guarantees of a tetrahedral discretization, leaving its accuracy sensitive to point density and to the statistical rayparity inside/outside test. Richer alternatives such as steady-state walk-on-spheres Monte Carlo estimators are a natural extension and could be plugged in behind the same interface in the future.

## S.2.6 Deriving Additional Annotations

Disparity maps. For stereo image pairs, a disparity map can be computed directly from the ground truth depth map if the camera parameters and the baseline are known. Assuming a rectified stereo setup with a baseline B and focal length $f _ { x }$ , the disparity d at a pixel with depth Z is simply given by $\begin{array} { r } { \dot { d } = \frac { f _ { x } \cdot B } { Z } } \end{array}$

Point maps. Point maps, which store the 3D coordinates $( X , Y , Z )$ of the scene point corresponding to each pixel, can also be derived from depth maps. These can either be obtained directly from the simulator—which leads to large file sizes but better precision due to fewer floating-point errors—or computed post-hoc by backprojecting the depth map. Specifically, given a pixel $( u , v )$ , depth Z, and camera intrinsics with focal lengths $f _ { x } , f _ { y }$ and principal point $( c _ { x } , c _ { y } )$ , the point coordinates in the camera’s frame are $\begin{array} { r } { X _ { c } = \frac { ( u - c _ { x } ) Z } { f _ { x } } , Y _ { c } = \frac { \ ' { ( v - c _ { y } ) Z } } { f _ { y } } } \end{array}$ , and $Z _ { c } = Z$ . Point maps are stored in the world frame, so this backprojection is followed by the rigid transform $[ X , Y , Z ] ^ { \top } = R [ X _ { c } , Y _ { c } , Z _ { c } ] ^ { \top } + { \bf t } .$ where R and t are the camera’s rotation and translation for that frame. Deriving point maps posthoc therefore requires the pose, the camera intrinsics, and the depth map; the simulator’s own point maps already apply this transform.

Scene flow. Scene flow can be roughly thought of as the 3D analog to optical flow. Instead of the two-dimensional motion field mapping pixel coordinates from one frame to the next, it captures the displacement of scene points in world coordinates.

By encoding scene dynamics, this three-dimensional vector field is particularly useful for navigation or robotics tasks, however, historically scene flow datasets have been underrepresented. One of the largest scene flow datasets is the Spring dataset [19], which relies on a custom patch of Blender to provide scene flow directly as a render pass.

While this is without a doubt the fastest way to get scene flow, as a by-product of rendering, we cannot make use of it in VisionSIM as it would require the framework to either maintain a custom branch of Blender or force users to compile the patched version themselves.

Instead, we opt to compute scene flow using other ground truth annotations which can all be generated using VisionSIM without relying on a custom Blender patch. This significantly reduces the complexity associated with distributing VisionSIM, but has two main downsides.

First, scene flow generation is now a post-hoc process, to compute forward flow, the ground truth depth of the next frame needs to be known, and vice-versa for backward flow. Since this process is not causal it has to lag the main rendering process. Currently, VisionSIM opts to compute scene flow exclusively after the render step is complete, to ensure dataset integrity in the event of an irrecoverable crash.

Second, to generate ground truth flow vectors, Blender needs to know ahead of time where the state of the world at the next (and previous) timestep. The implication is that both optical and scene flow cannot be generated when the user manually positions the camera within the render loop as shown in Fig. 1(C). Instead, the camera motion needs to be known pre-render, and communicated with Blender via the use of keyframes. In practice, this is seldom a limitation as VisionSIM provides utilities to set camera keyframes before invoking render animation.

To compute the forward scene flow $S _ { f }$ from time $t \tan t + 1$ we need the point maps $P M _ { t } , P M _ { t + 1 }$ and forward optical flow $F _ { f }$ for those times. We first get the world coordinates $( x _ { t } , y _ { t } , z _ { t } )$ for a pixel at time t using $P M _ { t }$ . Then, use $F _ { f }$ to warp the pixel coordinates from frame t to their corresponding locations in frame $t + 1$ , which yields $\left( u _ { t + 1 } , v _ { t + 1 } \right)$ . We sample $P M _ { t + 1 }$ at this location and obtain the corresponding world coordinates $( x _ { t + 1 } , y _ { t + 1 } , z _ { t + 1 } )$ . The difference between these two points in world coordinates is the desired forward scene flow.

## S.2.7 Interpolation

To accurately model high-speed sensors and artifacts such as motion blur, high frame rate data might be needed. While VisionSIM’s distributed architecture, equipped with its powerful keyframe remapping features, allows users to render datasets at arbitrary frame rates, in many cases, this is not needed. Every scene has a native frame rate, beyond which any motion or view-dependent effect becomes imperceptible, wasting additional rendering efforts.

In the real world, sampling the scene much faster than this intrinsic rate is beneficial. It reduces the observed noise and boosts the measured signal. This is especially important in the case of singlephoton cameras, which can run at a hundred thousand frames per second, yet each observation encodes very little information — a single bit’s worth in fact [48].

However, in simulation, we have access to perfect, noise-, and blur-free frames, so there is no need to render redundant data, it suffices to render fast enough and interpolate the rest of the way to reach our desired frame rate. This technique, to render ground truth at some slower rate and fill in the rest, can dramatically cut down on render times for scenes that have low native frame rates. The minimum frame rate at which to render ground truth frames depends entirely on the scene, its geometry, material properties, textures, camera parameters and resolution, and the speed of the scene’s relative motion. In general, the native frame rate is hard to estimate, even in simulations where all parameters are known.

Fortunately, precisely estimating this inflection point is usually not needed. Overshooting means that there will be extra overhead as we are required to render a few additional frames that are near duplicates. On the other hand, if we render slower than the native frame rate, adjacent frames will be more dissimilar. Thankfully, modern deep learning-based video interpolation methods can help fill this gap as they are specifically designed to interpolate between frames that encode large motion. In the limit, when using these methods on frames that have been rendered at or past the native frame rate, adjacent frames are identical and interpolation becomes trivial, effectively devolving into frame duplication. Conversely, when frames are too far apart interpolation will result in noticeable artifacts. Therefore, these interpolation methods give us some leeway in finding the best trade-off between rendering more ground truth frames, at the cost of additional rendering time, and interpolating between wider gaps in the data, which can lead to artifacts.

In Fig. 1 (B) we rendered a test trajectory at many different frame rates and in each case interpolated the resulting data up to a common frame rate of 6400 fps. We plot the average perceptual similarity between these interpolated frames and ground truth data (lower is better), as well as the total wall time to render and interpolate these different datasets as measured using a single rendering job, on an Nvidia RTX 3080. Notice that the time it takes to interpolate the frames is dwarfed by rendering time. The overall quality of the data improves with higher frame rates and lower interpolation factors, hitting an inflection point around 200fps, after which there are diminishing returns as rendering time explodes. From this, we can see that for this specific scene the native frame rate is around two to four hundred frames per second, and below this rate we incur significant interpolation artifacts, as shown. It takes just under 20 minutes to render this scene at 400fps and interpolate it 16× to 6.4 kHz, as compared to over four hours if we rendered every frame.

In practice, one would like to determine the optimal keyframe multiplier K that achieves this native frame rate automatically. Prior work such as ESIM [7] addressed this by dynamically triggering a new render whenever the per-pixel optical flow exceeds a fixed threshold. While theoretically appealing, this strategy is difficult to parallelize efficiently and produces non-uniform temporal sampling that is incompatible with many video interpolation methods and sensor modalities. We instead propose a simple yet highly effective closed-form iterative approach. Under the first-order approximation that the optical flow magnitude V scales as $K ^ { - 1 } - \mathrm { i . e }$ ., speeding up the frame rate by K reduces frame-to-frame motion proportionally — we can estimate the corrected multiplier $K _ { i + 1 }$ needed to scale the flows to a target value $\mathcal { V } _ { t a r g e t }$

$$
K _ { i + 1 } = K _ { i } + \alpha ^ { i } K _ { i } \cdot \left( \frac { \hat { \mathcal { V } } _ { q } ( K _ { i } ) } { \mathcal { V } _ { t a r g e t } } - 1 \right) _ { [ a , b ] } .\tag{S8}
$$

where $\hat { \mathcal { V } } _ { q } ( K _ { i } )$ is the measured q-th percentile flow magnitude at multiplier $K _ { i }$ , the subscript denotes clamping to the range [a, b] to prevent over or under shooting, and $\alpha ^ { i }$ acts as a dampening factor, with α typically set to 0.95. When there is no dampening or clamping, this equation reduces to directly scaling the keyframe multiplier by the multiplicative factor, which would — if flow is perfectly inversely proportional to $K -$ result in the target flow. The added terms help with convergence when this assumption breaks down.

![](images/22ea6ba6eb345f78e7c0c10e6dafc786a386b51ddc291dd41cbfd24383e815ae.jpg)

![](images/1588cd7e47bd2851800a278335657adfa075d520b752591581928d1450cb8653.jpg)

![](images/7c3985bd3a908747355404f6ef921519782ac9f0b131f0c1b36382f9d7c66a45.jpg)  
Suppl. Fig. 2: Convergence of the keyframe-multiplier search on the classroom scene. Each histogram shows the distribution of per-pixel optical flow magnitudes measured at a different iteration, with the target flow of 1 px. The 95th-percentile estimate $\hat { \mathcal { V } } _ { 9 5 }$ is shown as the dashed red line for each iteration. The first iteration overshoots, as many smaller flows are discovered to be around ∼0.2px, while the second iteration finds the optimal keyframe multiplier.

In this way the frame multiplier is set by the scene complexity and speed, through the optical flow proxy, as opposed to by the sensor we wish to emulate. We render at a rate that keeps the 95thpercentile optical flow at a few pixels and interpolate the remaining frames, so the interpolated sequence stays within the tolerance plotted in Fig. 1(B) of the fully rendered ground truth. Because the rate comes from the per-scene search below rather than a fixed multiplier, it adapts to each trajectory, and a single render supports emulation at any sensor frame rate above it.

To estimate $\hat { \mathcal { V } } _ { q } ( K _ { 0 } )$ , we need to render the optical flow of the full trajectory; however, this is much cheaper than rendering the full image frames. Not only can we speed up rendering by using a very low number of samples-per-pixel and disabling auxiliary passes, but we can also render at a proportionally smaller resolution and then scale up the flow fields accordingly. In practice, this technique converges to the desired operating point in just a few iterations, making it inexpensive to execute as a pre-processing pass before any full-scale rendering begins. As an example, on the scene in Fig 1 (B), this technique finds, in just two iterations, that to achieve a 95-percentile flow of less than two pixels we should use a keyframe multiplier of 4.16, meaning a native frame rate of 99.8fps, perfectly matching up with the inflection point on the plot. Rendering at this frame rate and interpolating up to the target rate of 6400fps reduces the total simulation time from 4 hours and 3 minutes to just under 4 minutes, a 60× speedup.

Fig. 2 illustrates the convergence of this procedure on the classroom scene [75] with a target flow magnitude of $\mathcal { V } _ { t a r g e t } = 1  { \mathbf { p } }  { \mathbf { x } }$ . Starting from an initial multiplier of $K _ { 0 } = 5$ , the measured 95thpercentile flow is $\hat { \mathcal { V } } _ { 9 5 } \approx 6  { \ p \mathrm { x } }$ — well above target, indicating the frame rate is too low. Following Eq. S8, the multiplier is updated to $K _ { 1 } = 3 0$ ; however, the flow distribution shifts and concentrates near 0.2 px, yielding a 95th-percentile of only 0.37 px, meaning we have overshot the target. A final update produces $K _ { 2 } = 1 1 . 2 $ , for which the 95th-percentile flow lands precisely at the 1 px target.

Finally, while this method produces substantial speed-ups with little impact on the quality for natural images, it does not work for all modalities. The input images need to be in-distribution for video interpolation methods to work well. Specifically, for non-tone-mapped, high dynamic range images, common methods such as RIFE [31] or FILM [76] do not work reliably. In these cases, we can either render every frame or, by also using ground truth depth maps, camera intrinsic, and pose, we can reproject pixels from one frame to a new one. This will cause a different type of artifact, namely bands of missing pixels due to parallax effects, which could be addressed using inpainting or colorization methods [77, 78].

## S.3 Physically accurate sensor emulation

At this stage, we have access to high-speed interpolated data, which we will need to further pro cess into the desired sensor modality. Specifically, now that world simulation is done, we perform sensor-realistic emulation for single-photon, high-speed, event, and conventional cameras, along with IMUs. Other camera types could be added by plugging in their image formation model on top of the simulated and interpolated data.

Conventional cameras: While rendered frames are good proxies for what a conventional camera might capture in ideal conditions, in many challenging real-world settings, these are significantly different, and accurate sensor modeling is required [79, 80].

The brightness of a conventional CMOS camera pixel can be modeled as a function of the scene radiance over the exposure time transformed by the camera’s optical stack and response function Γ, which encapsulates the camera response curve, tone mapping, quantization, and proprietary image signal processing [51]. Specifically, over an exposure time τ, a CMOS pixel will accumulate charge, and potentially saturate once it reaches its full well capacity (FWC). This process is subject to many sources of noise, including read noise and photon shot noise, which we absorb into N. The intensity I of the pixel can be modeled as:

$$
I = \Gamma \left( m i n \left( \int _ { \tau } \phi d t , { \mathrm { \bf ~ F W C } } \right) + \mathcal { N } \right) .\tag{S9}
$$

To emulate this, we sum together many interpolated intensity frames, discretizing the above integral and creating motion blur. It is important that this discretization be performed at a high frame rate otherwise, the motion blur will not be smooth and will lead to noticeable ghosting artifacts. We then add Gaussian read noise, apply a gain factor representing the sensitivity (ISO), and quantize to a certain bit-depth: the sensor model is adopted from the work of Hasinoff et al. [81]. More complex effects, such as optical aberrations or fixed pattern noise, can be added here as well.

The camera’s image signal processor (ISP) pipeline involves many computational steps after signal readout, such as de-mosaicing, de-noising, tone-mapping, and compression. The current version of the simulator provides baseline implementations for each of these steps (see methods). Future additions may include 3A algorithm implementations (auto-exposure, auto white-balance, and autofocus), and recent learning-based imaging pipelines (“neural ISPs”) that are optimized end-to-end.

High-speed cameras: Usually, also based on CMOS technology, high-speed cameras can be emulated similarly to conventional cameras. However, these suffer from low signal, as the integration time is short, and readout noise that can be an order of magnitude larger than their conventional counterparts<sup>1</sup>. Thus, we can use the same model, and adapt τ and N accordingly.

Stereo and multi-view cameras: Stereo cameras enable depth perception by capturing images from two viewpoints with a fixed baseline. We model stereo rigs by spawning two cameras with a fixed offset and intrinsically synchronizing their shutters. This allows for the generation of perfectly aligned stereo pairs and ground-truth disparity maps, which are essential for training and evaluating stereo matching algorithms [20, 19]. Multi-view cameras can be modeled similarly by spawning more cameras with different baselines and intrinsics.

Passive single photon cameras: Despite being excellent general purpose imagers, capable of extreme high-speed, low-light, and high-dynamic range imaging [82], single-photon cameras remain scarce and expensive. Emulating these cameras can help meet the need for cheap, high-quality, single-photon data and encourage future research.

Specifically, we emulate a single-photon avalanche diode-based camera, as these are rapidly becoming mainstream thanks to their CMOS-compatible fabrication process [83]. These are binary devices that do not suffer from read noise as they are entirely shot-noise-limited. However, SPC pixels reset after a photon detection event, during which no new photons can be detected. This effectively quantized the pixel’s output, leading it to follow a Bernoulli distribution parameterized by 1 − e<sup>−ϕτ</sup> [48], where ϕ is the scene’s radiant flux. By using the high-speed interpolated frames as a proxy for scene flux, we can emulate an SPC by sampling this distribution. Doing so accurately involves rectifying the tone-mapped frames to their linear intensities and accounting for other non-idealities such as dark count rates or low fill factors. See the supplement for more.

Active single-photon cameras: Robust depth perception using direct time-of-flight (dToF) sensors requires a high-fidelity, physically accurate model of an active single-photon camera (ASPC), also commonly known as single-photon LiDAR. We simulate the complete acquisition pipeline, mapping scene irradiance and illumination conditions to noisy photon timestamps. For each pixel, scene points within the FoV are accumulated and mapped to time-of-flight delays to form the scene re sponse function (SRF). Convolution with the instrument response function (IRF) then models the system’s temporal blur, while ambient illumination is treated as a uniform background rate. Together, these define the per-pixel photon arrival rate, from which discrete photon arrivals are drawn as an inhomogeneous Poisson process.

In practical single-photon LiDARs, not all photon arrivals are converted to photon detections due to sensor dead time. We explicitly model the detector’s dead time $\tau _ { d e a d }$ using a non-paralyzable dead time model where photons arriving within the recovery period are discarded. Our implementation can easily be extended to support paralyzable dead time. These functionalities accurately reproduce pile-up distortion, which is essential for emulating physically accurate LiDAR measurements and validating robust estimation algorithms. In addition to simulating raw photon timestamps, the simulator supports both standard Equi-width histogramming (EWH) methods and more recent Equi-depth histogramming (EDH) methods [84, 85].

Event cameras: By asynchronously emitting per-pixel brightness changes instead of full frames, event cameras are well-suited for many latency-critical applications such as object tracking [86, 87]. Our event model follows v2e [8]. Each pixel applies a probabilistic contrast threshold: a positive or negative event is emitted when the log-intensity change since the last event exceeds the threshold, with the threshold drawn stochastically per event to reproduce the threshold variation measured across a real sensor. The photodiode front end is modeled as a first-order low-pass filter, which sets the bandwidth limit that suppresses events under rapid illumination changes. A centre–surround spatial kernel applied to the pixel’s input reproduces the DVS contrast response. After an event, the pixel is held for a refractory period during which further changes are not reported. These stages take the interpolated high-frame-rate intensity sequence as input, which VisionSIM has already produced, so event emulation runs directly on both rendered and migrated video.

Inertial measurement units: Finally, we model IMUs as they are often co-located with many imaging sensors and are an integral part of many robotics and AR/XR applications where visualinertial or SLAM-based algorithms may be used. These contain an accelerometer and gyroscope that are modeled by a discrete-time random walk [88, 53, 89], which introduces both noise and a slow drift away from the camera’s actual linear acceleration and angular velocity, respectively. The reference values for both are estimated from the ground-truth camera pose sequence through standard numerical differentiation techniques. The noise standard deviations are set by default to values obtained in previous works in visual-inertial odometry [53], but may also be set differently using the datasheet of the sensor to be emulated.

Indirect time-of-flight LiDARs: Indirect time-of-flight (iToF) LiDARs estimate depth by measuring the phase shift or correlation between a modulated light source and the reflected signal [40]. Unlike dToF sensors that measure individual photon timestamps, iToF sensors typically modulate the active illumination with a periodic coding function—such as a sinusoid, square wave, or more exotic functions [41]—and record the integrated intensity over several correlation windows. Our simulator supports many common coding functions, enabling the study of different modulation schemes in challenging environments.

Thermal cameras: Unlike other modalities, long-wave infrared (LWIR) thermal cameras measure the radiation emitted by a surface due to its temperature. We emulate this by first rendering the temperature field produced by the world simulation’s heat-transport solver into in-band therma radiance using a physically motivated graybody emission model (Sec. S.5.4.1). We then model the temporal dynamics of the dominant commercial LWIR detector, the microbolometer. By filtering the rendered radiance to account for the detector’s thermal inertia, we accurately reproduce the characteristic lag and motion smear of real thermal video (Sec. S.5.4.2).

## S.4 Data storage and ML framework integration

Once the desired data has been generated, it needs to be stored in a convenient format that enables fast random access and iteration and easily integrates into downstream frameworks. While many such formats exist, we extend the data format informally introduced by NeRFs [90] to maintain compatibility with existing view synthesis and multi-view reconstruction frameworks such as NeRF-Studio [32] or COLMAP [91]. Specifically, we extend this format along two fronts: a sensoragnostic, and sensor-aware extension.

Both are based around maintaining a transforms.json file which encodes details about the camera’s intrinsic parameters, such as its resolution, focal length and principal point as well as a list of camera poses and asset paths for each instant of the camera trajectory. In its simplest form, these asset paths are just relative paths to the rendered image [90]. This format can be extended in a sensor-agnostic way, by allowing these asset paths to be more complex to account for other sensor modalities and ground truth annotations. For instance, events are encoded as a stream of (t, x, y, p) tuples encoding the time, x/y-coordinates, and polarity of the event, which are saved in a separate file that can be linked to.

For single-photon camera data, a sensor-optimized scheme is more appropriate since saving individual single-photon frames can use enormous amounts of storage as image codecs are poorly suited at encoding such highly stochastic data [92]. Further, many applications, such as Gaussian Splatting [93], ingest data at the granularity of pixels instead of full frames, which would require decoding a whole binary frame to sample just a few pixels. This is not an issue when there are only a few hundred images as pixel values can be pre-loaded and cached on the GPU for fast training [94], but we quickly run into memory limitations if we attempt to pre-load potentially millions of single photon frames.

To overcome these issues, we store all SPC data as a single large bitpacked numpy array. In this manner, the data is losslessly compressed by 8× and can be memory mapped which enables fast random access to the whole dataset. Since the minimum addressable unit of memory is a byte, we are forced to read in a whole byte, whether it represents eight pixels or only one. Thus this compression scheme has virtually no runtime costs, and decompression can be performed on the fly when accessing pixels. VisionSIM abstracts away the tricky bitmasking operations required to perform this compression, instead exposing a high-level API to read and write SPC data directly.

For both these formats, VisionSIM provides data-loading utilities that can directly be used with modern deep learning frameworks such as PyTorch or TensorFlow to train machine learning models.

## S.5 Forward models and implementation details for selected sensors

## S.5.1 Emulating single-photon cameras

Assuming constant flux during the exposure time, the number of photons incident on a pixel during an exposure time τ follows a Poisson distribution parameterized by the total incident light during that exposure, namely ϕτ, where ϕ is the scene’s radiant flux. A SPAD-based SPC does not suffer from read noise; however, its pixels reset after a photon detection event, during which no new photons can be detected. This quantizes the pixel’s output, leading it to follow a Bernoulli distribution with parameter $1 - e ^ { - \phi \tau }$ [48].

Therefore, SPC can be emulated by Bernoulli sampling the interpolated intensity frames. However, the ground truth frames do not encode scene flux; they encode the scene’s intensity, often non linearly. We must first ensure the frames are in a linear colorspace, either by generating them as such or undoing any tone mapping that may have been applied. This intensity then needs to be mapped to scene flux; however, up until now, there has been no concept of scene brightness in absolute terms. While a user can set the power of every light source in Blender, this does not translate directly to physical units. Instead, we introduce a correction factor f that maps a pixel’s linear intensity to a flux value, absorbing into it any non-idealities such as dark count rates or low fill factors.

## S.5.2 Active single-photon camera modeling

Robust depth perception using direct time-of-flight (dToF) sensors requires training data that captures the nuances of real-world hardware. To this end, our simulator implements a high-fidelity model of an Active Single-Photon Camera (ASPC) that goes beyond the noise-free depth abstractions of standard game engines. We simulate the complete acquisition pipeline, mapping scene irradiance to noisy photon timestamps, while also modeling on-sensor compression schemes critical for resource-constrained applications like autonomous robots and AR/VR headsets. We discuss the forward imaging model in Section 4.1. In this section, explain it in detail along with the implementation details.

## S.5.2.1 Forward model implementation details

The forward imaging model can be divided into three main stages: (i) computing the photontransient, which models the geometric and radiometric scene properties, (ii) computing the photon arrival rates, which models the active and ambient light sources, and (iii) computing the LiDAR measurements, which models pileup distortion, compressed readouts, the effect of TDC resolution, and synchronous or free-running mode of operation of SPAD sensors.

## Modeling the photon-transient

The photon-transient T is the scene response function (SRF) modeling the interaction of an ideal laser pulse with the scene. In VisionSIM we utilize ground-truth data (RGB albedo and depth maps) generated by the rendering engine. To bridge the discrete spatial domain of the renderer and the temporal domain of the SPAD, we employ a per-pixel Field-of-View (FOV) integration method. For every sensor pixel $( u , v )$ , we accumulate the weighted contribution of scene points within its projected cone. The simulator defines a weighted per-pixel FOV mask, $\mathcal { M } _ { u v }$ , which encapsulates the geometric relationship between the scene and the sensor optics<sup>2</sup>. For each pixel, the discrete scene response $T [ b ]$ at bin index b aggregates the radiometric contributions of scene points within a field-of-view mask $\mathcal { M }$ as

$$
T [ b ] = \eta _ { \mathrm { o p t } } \sum _ { p \in \mathcal { M } } \mathbb { I } \big ( b \Delta t \leq t _ { p } < ( b + 1 ) \Delta t \big ) \frac { \alpha _ { p } } { z _ { p } ^ { 2 } } W ( p ) ,\tag{S10}
$$

where $\Delta t$ is the temporal bin width, $\alpha _ { p }$ is albedo, $1 / z _ { p } ^ { 2 }$ the inverse-square falloff, $W ( p )$ the mask weight, and $\eta _ { \mathrm { o p t } }$ the quantum efficiency. This mask-based approach allows us to simulate realistic lens effects, such as vignetting and blooming, which are critical artifacts in robotic perception pipelines.

## Modeling the active and ambient light sources

The raw scene transient assumes an ideal impulse illumination. To model the active source, we convolve the transient with the system’s Instrument Response Function (IRF) represented by g(t). The simulator supports parametric pulse shapes (e.g., Gaussian, Square) or empirical IRFs derived from hardware calibration. Robotic platforms often operate in uncontrolled lighting $( \mathrm { e . g . }$ , direct sunlight). We model ambient light not as additive noise, but as a distinct photon source. The simulator configures spectral bandpassfilters, ensuring that only ambient light overlapping with the laser’s spectrum contributes to the background photon count rate, modeled as a DC offset $\lambda _ { \mathrm { a m b i e n t } } .$ The resulting computation represents the expected photon arrival rate $\lambda ( t )$ incident on the detector:

$$
\lambda ( t ) = ( T * g ) ( t ) + \lambda _ { \mathrm { a m b i e n t } }\tag{S11}
$$

## Modeling sensor properties and compressed readouts

To transform continuous photon rates into discrete digital measurements, VisionSIM models the stochastic behavior of sensor electronics along with realistic hardware readout schemes. First, photon arrivals are sampled from an inhomogeneous Poisson process, where the detected count k within a given time bin is distributed as $k \sim \mathcal { P } ( \lambda ( t ) )$ to account for both active signal and ambient background flux. We explicitly incorporate a non-paralyzable dead time $( \tau _ { d e a d } )$ model to simulate the non-linear pileup distortions typically observed when operating in high-flux scenarios. We support two different on-sensor compression methods: (i) standard equi-width histogramming (EWH), which aggregates arrivals into fixed temporal intervals, used in conventional LiDARs, and (ii) equi depth histogramming (EDH). Designed to mitigate bandwidth constraints in high-resolution arrays, EDH utilizes adaptive quantization to dynamically vary bin widths based on local photon density, preserving high temporal resolution for sharp, peaky distributions while compressing sparse background noise. Suppl. Fig. 3 shows the effect of different readout schemes, varying FOV mask, sensor temporal resolution and sensor dead time.

## PyTorch implementation with unit-aware configuration system

The simulator is implemented entirely in PyTorch, prioritizing modularity and vectorized operations. By representing the sensor measurements as a high-dimensional tensor, we leverage massive parallel operations. Unlike CPU-bound simulators that iterate over pixels, our vectorized implementation processes mega-pixel arrays simultaneously on the GPU, significantly reducing the simulation time for dense SPAD arrays.

Unlike rigid legacy simulators, our framework utilizes a strict, unit-aware configuration system (powered by pint and ruamel.yaml). This ensures that all simulation parameters, from laser power (Watts) to pixel pitch (Micrometers), are dimensionally consistent, eliminating physical unit errors before runtime.

Another distinguishing feature of our framework is the fully differentiable execution modes, which use PyTorch’s autograd functionality. This formulation allows gradients to flow from the downstream task loss (e.g., depth reconstruction error) back to the sensor parameters. This enables endto-end optimization of hardware parameters, such as the sensor IRF or dead time.

## S.5.2.2 Physically Constrained Single Photon LiDAR Simulations using VisionSIM

Our goal with VisionSIM is to bridge the sim-to-real gap, and to do so, we need to be able to model sensor physics realistically. The reason conventional simulators fail to do this is that they create ideal, high-resolution depth maps from scene data. They fail to model limitations in sensor performance as we show in Fig. 3 (A, B, C, D). The tradeoff between bandwidth and resolution is something every single-photon LiDAR designer needs to consider when developing a system. Lowering spatiotemporal resolution offers a balance between the data bottleneck and fidelity. However, to evaluate the design choices, conventional simulators use techniques such as artificial quantization and resize-interpolation, which do not adequately model the physical world. VisionSIM enables physically accurate emulation of realistic sensor and optical properties. Complex optical effects can be modeled using per-pixel FOV masks, shown in Fig. 3 (E), including vignetting, a process largely ignored by conventional simulators. The FOV masks also allow us to emulate sensors of varying spatiotemporal resolution while retaining a physically accurate acquisition model (Fig. 3 F, H).

VisionSIM includes environment non-idealities during acquisition, such as noise from ambient light (Fig. 3 G) as well as non-idealities unique to this modality. One example of this stems from the implications of dead time. This is a requirement of these sensors because their ability to detect single photons stems from their high sensitivity to energy introduced to the system. They require time to return to an equilibrium state before additional detections. An unavoidable problem arises from this physical process, which is known as pileup. Pileup occurs when photons are not detected during a SPAD’s dead time, which leads to two problems. Firstly, photon counts are underestimated. Secondly, these early detections bias the sensor towards earlier timing bins and can cause a reported depth to be lower than an object’s true depth (referred to as range-walk error) as demonstrated in Suppl. Fig. 3.

Conventional sensor simulators fail to bridge the Sim-to-real gap  
![](images/f49b290526e029b965d47c20612d8bf86d3326b3edd6651e5b2a5b7a55ef75f8.jpg)

![](images/c528d01c735ecc4b31a96b9d3fb67ca0244af38310edd4fa104473befb764efb.jpg)

![](images/e6bb1c54afcd9a454d820489a674887088156abd6a361654825527e6c1a3cc1e.jpg)

![](images/9a35461aaf4b522df697a54d61cc34f277ded332e80e97c5c0008c3e313e157a.jpg)  
Physically accurate sensor simulators enabled by VisionSim  
(G) High spatial res | high temporal res | high sensor noise

![](images/72e22026d39ec497cafcd673c1cfc364193025c59a5f4a6e758d3d51f7b5db3c.jpg)

![](images/3137be671ffbaaaab350a8b158af790c494b8d5f572c96ccda8d44f02f678963.jpg)

![](images/f33fcb0adafb9f89e1418bb8ad6122a076184dc83a51a7b60f3ea48f73eab95d.jpg)

![](images/a6fa83fb3c706479f835e842a5c10d0e4e6b2f5fedad89eb782dc15605095728.jpg)

![](images/73543d422d826e13f7f9d04e8fe5ad0d89e6bb4d592e4190461920be1abbda4b.jpg)  
Suppl. Fig. 3: VisionSIM enables physically accurate 3D sensor simulations: Conventional simulators create idealistic depth maps (B) from scene data (A). They fail to bridge the sim-to-real gap, ignoring sensor physics by emulating low temporal resolution (C) and low spatial resolution (D) via naive quantization. VisionSIM enables physically accurate emulation of sensor properties such as per-pixel FOVs (E); high spatial resolution with low sensor noise (F); dead time effects (G) and hardware constraints, including low spatial (H) and temporal (I) resolution.

## S.5.3 Indirect Time-of-Flight Simulation

Unlike direct time-of-flight (dToF) systems that measure the timestamps of individual photon detections, indirect time-of-flight (iToF) cameras estimate depth by emitting amplitude-modulated light and analyzing the phase shift of the reflected signal relative to the source [40]. An iToF system consists of a temporally modulated light source with radiant intensity $M ( t )$ and a sensor with a temporally modulated exposure (demodulation) function $D ( t )$ . Let $\Gamma ( p )$ be the scene depth at pixel $p .$ The radiance $L ( p , t )$ incident at pixel p at time t is given by [41]:

$$
L ( p , t ) = \beta ( p ) M \left( t - \frac { 2 \Gamma ( p ) } { c } \right) + L _ { a } ( p ) ,\tag{S12}
$$

where $c$ is the speed of light, $\beta ( p )$ encapsulates the scene albedo, reflectance, and inverse-square fall-off, and $L _ { a } ( p )$ is the ambient light component.

The brightness $B ( p )$ at pixel p over an integration time $\tau$ is the correlation between the incoming radiance and the demodulation function:

$$
B ( p ) = \int _ { 0 } ^ { \tau } D ( t ) L ( p , t ) d t = \beta ( p ) F ( \Gamma ) + A ( p ) ,\tag{S13}
$$

where $\begin{array} { r } { A ( p ) = L _ { a } ( p ) \int _ { 0 } ^ { \tau } D ( t ) d t } \end{array}$ is the ambient component. $F ( \Gamma )$ is the normalized correlation function between the modulation and demodulation functions:

$$
F ( \Gamma ) = \frac { 1 } { M _ { t o t a l } } \int _ { 0 } ^ { \tau } D ( t ) M \left( t - \frac { 2 \Gamma ( p ) } { c } \right) d t ,\tag{S14}
$$

where $\begin{array} { r } { M _ { t o t a l } = \int _ { 0 } ^ { \tau } } \end{array}$ M(t)dt is the total flux emitted by the source. Finally, the measured brightness $\tilde { B } ( p )$ is corrupted by shot noise, quantization and other non-idealities as governed by the sensing technology.

By varying the modulation and demodulation functions across multiple phase offsets, the sensor captures several correlation measurements, commonly referred to as taps, to reconstruct the depth $\Gamma ( p )$ . The top row of Fig. 2 illustrates this acquisition process under ideal conditions. Here, the emulated iToF camera is using a conventional 3-tap delta-sinusoidal modulation/demodulation scheme (others such as square, ramp, Hamiltonian, and Hilbert [41] are also available within VisionSIM) with a modulation frequency of 20 MHz, showing the ground truth RGB scene, the corresponding ground truth depth map, and the ideal correlation measurements across three phase taps.

However, real-world iToF systems are highly susceptible to physical non-idealities and environmental factors that conventional computer vision simulators fail to model. VisionSIM addresse this gap by implementing physically accurate emulation of these effects, enabling the validation of downstream depth estimation algorithms under realistic conditions. As shown in the bottom row of Fig. 2, we simulate several critical iToF failure cases and demonstrate their direct impact on depth reconstruction. All examples in the figure are the result of varying a single simulation parameter, and these effects can of course be combined:

• Incorrect Modulation Frequency: The modulation frequency defines the unambiguous depth range of the sensor. In the figure, the incorrect modulation frequency uses 35 MHz (yielding an unambiguous depth of only ∼4.3 meters, which is not enough for the scene). The phase wraps around, causing severe phase wrapping ambiguities where distant objects appear deceptively close.

• Sensor Saturation: When the exposure time is too long, or an object is highly reflective or positioned too close to the active light source, the accumulated charge can exceed the sensor’s full well capacity. This saturates the pixel and corrupts the correlation measurements. Here, clipping occurs because the full well capacity of 50k electrons is exceeded. This injects nonlinear distortions that corrupt the reconstructed depth (notice the front row chairs and ceiling lamps).

• Ambient Light Corruption: Strong background illumination introduces a significant DC off set to all taps. In this example, the ambient light level is raised to match the power of the active source. This degrades the modulation contrast and signal-to-noise ratio, leading to incorrect depth reconstruction, particularly for distant objects as they are more dimly illuminated by the active source.

• Motion Blur: Because iToF systems must sequentially capture multiple frames (taps) to reconstruct a single depth map, any relative motion between the camera and the scene during the acquisition sequence leads to multi-tap motion artifacts. For this example, motion blur happens over 100ms. VisionSIM simulates this effect by temporally offsetting the frames used for each tap, accurately reproducing the resulting depth tearing and ghosting at object boundaries.

• Sensor Noise and Quantization: We model physical noise sources such as shot noise and read noise, alongside the finite resolution of the analog-to-digital converters (quantization). In the figure, the noise corresponds to shot noise plus a read noise of $1 0 e ^ { - }$ and a 12-bit ADC. These factors lead to high-frequency depth noise and quantization banding in the reconstructed depth map.

By modeling these physical degradation processes, VisionSIM provides a reliable playground for developing and stress-testing robust phase-unwrapping, de-noising, and high-dynamic-range iToF depth reconstruction pipelines.

## S.5.4 Thermal Cameras

Long-wave infrared (LWIR, 8–14 µm) thermal cameras occupy a fundamentally different place in the imaging stack than the other modalities in VisionSIM: rather than measuring light scattered off the scene within the last microseconds, they measure the long-wave radiation a surface emits as a consequence of its temperature. The world simulation layer supplies that temperature field through the integrated heat-transport solver (Sec. S.2.5.2); here we describe how the sensor emulation layer turns it into a measurement in two stages: rendering the temperature field into in-band thermal radiance with a graybody shader (Sec. S.5.4.1), and applying the microbolometer image-formation model whose thermal inertia gives real LWIR video its characteristic lag and smear (Sec. S.5.4.2).

## S.5.4.1 Rendering the Temperature Field into Thermal Radiance

The solver produces a per-vertex temperature field, which is not yet an image. To turn it into the in-band radiance a thermal camera would collect, we render the scene with a physically motivated graybody emission model. Rather than maintaining a separate thermal copy of the scene, we render thermal imagery from the same scene, camera, and animation as the RGB render by toggling a small set of scene-level overrides—swapping each material for its thermal counterpart, and by disabling visible-band lamps which an LWIR camera does not see. This guarantees that the RGB and thermal images are pixel-aligned and that any edit to the scene propagates to both modalities.

Graybody emission shader. At LWIR wavelengths each surface radiates as an opaque graybody, contributing both its own thermal emission within the camera’s sensitivity band and a reflection of the long-wave radiation arriving from the rest of the scene. We model the outgoing in-band radiance at x as

$$
L _ { \mathrm { o u t } } ( { \bf x } ) = \epsilon ( { \bf x } ) { \cal U } \big ( T ( { \bf x } ) \big ) + \big ( 1 - \epsilon ( { \bf x } ) \big ) L _ { \mathrm { i n } } ( { \bf x } ) ,\tag{S15}
$$

where $T ( \mathbf { x } )$ is the surface temperature from the heat solver, $\epsilon ( \mathbf { x } ) \in [ 0 , 1 ]$ is the LWIR emissivity, $L _ { \mathrm { i n } } ( \mathbf { x } )$ is the incident long-wave radiance gathered from the rest of the scene [95], and $U ( T )$ is the camera response to a blackbody at temperature T, obtained by integrating Planck’s law against the camera’s spectral response and generally well-approximated by the Sakuma–Hattori model [96, 95]. The first term is the graybody emission within the LWIR specturm, scaled by emissivity through Kirchhoff’s law; the second is its conjugate reflection, weighted by $1 - \epsilon ,$ , which is exactly the reflectivity an opaque graybody assigns by energy conservation. Although the angular reflectance of real materials in the $8 - 1 4 \mu \mathrm { m }$ band is not as well characterized as in the visible, we adopt a Lambertian BRDF for the reflected term: it is the canonical choice for a graybody, keeps the model fully determined by the single per-surface parameter $\epsilon ,$ and is straightforwardly replaceable by a measured wavelength-specific BRDF as the literature matures. The two terms together capture the qualitative behavior of real LWIR imagery without any material-specific special cases: a polished low-ϵ metal “looks cold” because its mix weights the reflection of its cooler surroundings, while a high-ϵ matte surface reads its own emitted light almost directly.

## S.5.4.2 Microbolometer Thermal Inertia

A real uncooled LWIR camera does not record the instantaneous in-band radiance produced by the renderer. The dominant commercial LWIR detector is the microbolometer: an array of thermally isolated absorber membranes, one per pixel, whose temperature is set by the incident radiation and read out as an electrical resistance. Because the membrane has finite heat capacity, its temperature cannot change instantly with the incident radiation; the readout instead lags behind, and that lag is what gives real thermal video its characteristic motion smear and ghost trails behind moving hot objects [97].

Quantitatively, a single absorber membrane is a lumped thermal mass connected to a substrate at temperature $T _ { s }$ through legs of finite thermal conductance, obeying the first-order ODE [97, 98, 99]

$$
\frac { C _ { \mathrm { t h } } } { G _ { \mathrm { t h } } } \frac { \mathrm { d } \Delta T ( t ) } { \mathrm { d } t } + \Delta T ( t ) = \frac { \alpha _ { \mathrm { a b s } } } { G _ { \mathrm { t h } } } \phi ( t ) ,\tag{S16}
$$

where $\Delta T ( t ) = T _ { m } ( t ) - T _ { s }$ is the membrane-to-substrate temperature difference, $C _ { \mathrm { t h } }$ is the membrane heat capacity $( \mathrm { J } / \mathrm { K } ) , G _ { \mathrm { t h } }$ is the thermal conductance of its supports $( \mathrm { W / K } ) , \alpha _ { \mathrm { a b s } }$ is the membrane’s in-band absorptivity, and $\phi ( t )$ is the instantaneous incident in-band radiative power (W). The ratio $\tau = C _ { \mathrm { t h } } / G _ { \mathrm { t h } }$ has units of time and is the membrane’s thermal time constant, the natural timescale on which it equilibrates to a change in incident radiation; for commercial uncooled LWIR detectors, τ typically lies in the 5–15 ms range. Defining the hypothetical instantaneous steady state $T _ { m } ^ { \mathrm { s s } } ( t ) = T _ { s } + \left( \alpha _ { \mathrm { a b s } } / G _ { \mathrm { t h } } \right) \phi ( t )$ —the temperature the membrane would reach if it equilibrated infinitely fast to the radiation present at time $t \mathrm { - E q }$ . S16 takes the canonical first-order form

$$
\tau \frac { \mathrm { d } T _ { m } ( t ) } { \mathrm { d } t } + T _ { m } ( t ) = T _ { m } ^ { \mathrm { s s } } ( t ) .\tag{S17}
$$

The interpretation is direct: the membrane temperature $T _ { m } ( t )$ chases its instantaneous target $T _ { m } ^ { \mathrm { s s } } ( t )$ at a rate set by τ. The readout integrated circuit then reports a digital count $I ( t ) = P _ { 1 } T _ { m } ( t ) { + } P _ { 2 }$ that is affine in $T _ { m }$ , so up to a one-time factory calibration the reported video is a temporally low-passfiltered version of the scene radiance. Crucially, $T _ { m } ^ { \mathrm { s s } } ( t )$ is exactly the idealized infinite-bandwidth measurement that a perfect (zero-τ) camera would produce—which is precisely the in-band radiance $L _ { \mathrm { o u t } }$ that our graybody renderer writes into each frame (Eq. S15). We can therefore synthesize a realistic microbolometer response by filtering the rendered radiance sequence through Eq. S17.

Closed-form solution and discretization. Equation S17 is a linear first-order ODE; multiplying by the integrating factor $e ^ { t / \tau }$ and integrating yields the convolution of the ideal radiance against a one-sided exponential kernel [97],

$$
T _ { m } ( t ) \ : = \ : \frac { 1 } { \tau } \int _ { - \infty } ^ { t } e ^ { - ( t - s ) / \tau } T _ { m } ^ { \mathrm { s s } } ( s ) \mathrm { d } s .\tag{S18}
$$

Equation S18 says that the current membrane temperature is a weighted average of all past radiance, with weights given by a decaying exponential of timescale τ: recent radiance contributes strongly, while radiance from many τ ago has been “forgotten.” This is the precise sense in which the bolometer acts as a temporal low-pass filter on the scene.

To apply this to rendered video, we discretize Eq. S18 at the frame rate. Let $t _ { n } ~ = ~ n \Delta t$ with $\Delta t = 1 / \mathrm { f p s }$ , let $x [ n ] : = T _ { m } ^ { \mathrm { s s } } ( t _ { n } )$ denote the rendered input at frame $n ,$ and let $y [ n ] : = T _ { m } ( t _ { n } )$ denote the filtered output we want to compute. Splitting the convolution integral at the previous frame time $t _ { n - 1 }$ and factoring the exponential kernel as $e ^ { - ( t _ { n } - s ) / \tau } = e ^ { - \Delta t / \tau } e ^ { - ( t _ { n - 1 } - s ) / \tau }$ gives

$$
y [ n ] \ = \ e ^ { - \Delta t / \tau } y [ n - 1 ] \ + \ \frac { 1 } { \tau } \int _ { t _ { n - 1 } } ^ { t _ { n } } e ^ { - ( t _ { n } - s ) / \tau } T _ { m } ^ { \mathrm { s s } } ( s ) \mathrm { d } s ,\tag{S19}
$$

where the first term is exactly what the convolution from $- \infty \mathrm { ~ t o ~ } t _ { n - 1 }$ contributes, attenuated by one frame’s worth of exponential decay. The renderer samples the scene once per frame with no within-frame motion blur, so we model the input as piecewise-constant across each interval (a zeroorder hold), $T _ { m } ^ { \mathrm { s s } } ( s ) \approx x [ n ]$ for $s \in \mathsf { \Gamma } ( t _ { n - 1 } , t _ { n } ]$ . Pulling x[n] out of the integral and evaluating $\begin{array} { r } { \frac { 1 } { \tau } \int _ { 0 } ^ { \Delta t } e ^ { - u / \tau } \mathrm { d } u = 1 - e ^ { - \dot { \Delta t } / \tau } } \end{array}$ collapses Eq. S19 to a single-pole infinite impulse response low-pass filter,

$$
y [ n ] \ = \ a y [ n - 1 ] + \left( 1 - a \right) x [ n ] , \qquad a = e ^ { - \Delta t / \tau } .\tag{S20}
$$

For piecewise-constant input this is the exact integrator of Eq. S17, not a finite-difference approximation, and the only free parameter is the single dimensionless ratio $\Delta t / \tau$ that determines $a .$ . The limits are sanity-checking: $\tau  0$ (instant camera) gives $a  0$ and ${ \dot { y } } [ n ] = x [ n ]$ , so the output equals the input with no lag; $\tau \to \infty$ (infinitely sluggish camera) gives $a \to 1$ and $y [ n ] = y [ n - 1 ]$ so the output is frozen. In the regime of interest, at 60 fps and $\tau = 1 0 \mathrm { m s } , a \approx 0 . 1 9$ and a hot pixel reaches $\sim 9 0 \%$ of its steady response in about three frames; larger τ stretches the lag proportionally. When $\tau \ll \Delta t$ (a fast detector under a slow render) the zero-order hold becomes inaccurate, and we instead linearly interpolate the input between rendered frames and run $S$ substeps per frame with $a ^ { \prime } = e ^ { - \Delta t / ( S \bar { \tau } ) }$ , recovering $\mathrm { E q . } \ : S 2 0 $ at $S = 1$

Scope of the thermal sensor model. Our graybody shader (Eq. S15) assumes a Lambertian BRDF and a single scalar emissivity, whereas real materials exhibit angularly and spectrally varying reflectance in the 8–14 µm band that is far less characterized than in the visible. On the detector side, our current model captures only the microbolometer’s thermal inertia, the first-order temporal lowpass that produces motion smear, and not the temporal noise families that define the NETD [100], the spatial fixed-pattern and column-stripe structure, the focal-plane-array drift between flat-field corrections, vignetting, or narcissus-like self-emission; some of these could be layered onto the rendered radiance as a post-process following the standard sensor signal chain, which we leave to future work.

## S.6 Building Digital Twin of Commercial dToF LiDAR

To demonstrate VisionSIM’s ability to bridge the real-to-sim gap we conducted an experiment in which we implemented a perceptual digital twin and verified its accuracy with respect it’s real world counterpart (as described in Sec. 5).

## S.6.1 Capture Setup

For this evaluation we captured the real scene using an off-the-shelf, commercial, single photon LiDAR developed by ADAPs Photonics [43] (the ADS6311) used because it gives access to raw unprocessed histograms. The ADS6311 is a 1D scanning LiDAR with a resolution of 256 × 192 with 3 × 3 spatial binning to improve SNR (i.e. there are 768 × 576 SPAD elements). This device has a dead-time of ∼11ns and a temporal resolution of 754ps. The transmitter uses VCSEL laser diodes to scan a 6-row-wide band in the vertical direction.

We set up a scene in lab with a retroreflective stop sign. Retroreflector’s send incident light back towards its source resulting in two phenomena: pileup and glare (referred to as bloom in the LiDAR community). Pileup is the result of a SPAD element’s ”dead-time” which biases the sensor toward earlier detections. In the event of strong returns such as those from a retroreflector, this bias is strengthened which leads to what is referred as range walk error (an error in the measured depth) as well as a reduction in measured intensity (due to photons in later time bins being missed). In addition to causing pileup, some of the returning waveform scatters diffusely on the lens as well as bouncing between the lens, receiver housing, and pixel array which all lead to spurious detections. This is the cause of what is known as glare or bloom (in traditional CMOS imaging this is sometimes called lens flare).

This evaluation gave us the ability to determine if VisionSIM could faithfully reconstruct the artifacts stemming from the physical acquisition process which is demonstrated in Figure 4.

## S.6.2 Blender Simulated Scene

We then reconstructed this scene in Blender with the stop sign and background at roughly the same depth and dimensions. Also, with the camera at the same position as it was when we captured the real scene. This is not a perfect reconstruction as the ADAPs kit provides only opencv-style coefficients which Blender does not follow for their barrel distorted lenses. Thus, we used a lens and camera model that was a perceptually similar approximation to the ADS6311. Additionally, currently VisionSIM does not directly support retroreflective objects; to overcome this we used the material identifiers and manually set albedo to approximate the extremely high albedo of retroreflectors (as well as the comparatively low albedo of the background, a black canvas tarp).

## S.6.3 Learning the Impulse Response Function

To evaluate VisionSIMs fully differentiable pipeline, we started with an initial assumption of an impulse response function that is a perfect delta (with some small positive constant in the remaining bins to avoid zero gradients). For learning the IRF we used a single scene that just contained the background tarp. In practice, a more controlled setup with known albedo targets should yield even better results. We compared a simulated histogram to the true measured histograms using a crosscorrelation loss function (over 1000 epochs) given by:

![](images/1fc8b17ac30a593a9d80916827316bc216a85ee364a0f1528c99d93cba58023f.jpg)  
Suppl. Fig. 4: Error Maps for digital twin evaluation; inset: RMSE between the measured depth map and the simulated depth map.

$$
C o r _ { l o s s } = \mathrm { m e a n } ( 1 - \mathrm { m a x } \left( p r e d * g t \right) ) + \lambda T V ( b _ { i } - b _ { i + 1 } )\tag{S21}
$$

where $p r e d * g t$ is the pixel-wise cross-correlation between the predicted (simulated) histogram and measured histogram across a $2 0 \times 2 0$ patch, and the total variation term is a smoothness penalty between adjacent bins to avoid sharp spikes in the IRF. The comparison given in Figure 4 shows the measured histogram as compared to a simulated histogram using our learned IRF for a point on the stop sign and a point in the region containing glare artifacts.

## S.6.4 Comparison to Real Capture Details

To simulate the glare artifacts we measured the PSF function of the ADS6311 by disabling the transmitter and pointing an IR flashlight at the same wavelength toward the receiver. We used a small aperture and baffle to ensure light was incident on a single pixel and that there were no multipath effects being measured respectively. These 49 measured PSF’s are treated as a basis, with remaining positions interpolated by a distance-weighted sum of the 49 measured positions. We plan to add more robust lens modeling that includes a parametric form of this glare model and can be learned in a similar fashion to the IRF learning described in section S.6.3. Error maps for each additional modeled effect (IRF, pileup, and bloom) can be found in Figure 4. It should be noted that the discrepancy between the reported RMSE from the main paper and this figure is due to a tighter crop around the region of interest. This is in part to highlight the effects of our realistic modeling but also because the background tarp was the most challenging to match in Blender, causing error unrelated to VisionSIMs physical model but instead due to imperfections in how the ground truth depth was calculated for the tarp in Blender.

## S.7 Dataset generation interface

Asset curation. To help accelerate research, we release all 3D assets and scripts used to create sensor-realistic datasets, lowering the need for manual asset curation, which can be a significant hurdle. Specifically, we release the 100 high-quality rigged scenes, some of which are shown in Fig. 5, which have been used to create the VisionSIM dataset. These scenes have mostly been crowdsourced from sites such as BlendSwap [101] or Blender’s example scenes [75], although some have been made or heavily modified specifically for this use case. All scenes are fully rigged, physics-baked, and validated to render without modification using the Blender version pinned to the corresponding VisionSIM release. These assets are released under the Creative Commons license (CC BY 4.0) or similar permissive licenses.

Generation pipeline. Dataset generation is driven by a single script, scripts/mkdataset.py, that wraps the same RenderConfig dataclass and render animation backend used throughout VisionSIM’s CLI. A user specifies a directory of blend-files and an output directory; the script then (i) uniformly samples non-overlapping sub-trajectories from the animation range of each scene, or renders the whole trajectory if desired, (ii) dispatches each (scene, trajectory) pair as an asynchronous render job via BlenderClients.pool, and (iii) saves every sequence under a deterministic path which also contains the explicit configuration used to produce the sequence. All relevant metadata are stored in per-modality SQLite databases alongside the raw data. The RenderConfig dataclass exposes every relevant parameter — resolution, render device, keyframe multiplier, types of ground truth, noise threshold, and per-modality flags — entirely through the CLI, so a full reconfiguration of the dataset requires no code changes. For example, to render a 800 × 800 dataset at 100 fps with depth, normals, optical flow, and segmentation using five parallel jobs:

![](images/683fa7953de2add8599e8ef630ac6918a5859541560af7523e9ec1a69b194f18.jpg)  
Suppl. Fig. 5: Example scenes from the VisionSIM dataset. A collection of diverse indoor scenes featuring different lighting conditions, materials, and ground truth annotations.

python scripts/mkdataset.py create-datasets \

--scenes-dir=scenes/ --datasets-dir=datasets/ \

--sequences-per-scene=1 \

--render-config.width=800 --render-config.height=800 \

--render-config.depths --render-config.normals \

--render-config.flows --render-config.segmentations \

--render-config.keyframe-multiplier=4.0 \

--render-config.jobs=5

For the full dataset, this script was submitted to HTCondor [65] via a containerized VisionSIM image, distributing work across thousands of CPUs and hundreds of GPUs as described in Appendix S.2. Sensor emulation (interpolation, SPC, event camera, iToF, thermal, IMU) is applied as a post-processing pass on the ground-truth renders using the same VisionSIM library functions described in Appendix S.3, each of which has a corresponding command-line interface tool.

Reproducibility. Every VisionSIM release is containerized with Docker [66] and tested end-to-end in CI, ensuring that a pinned version of the library and its container can regenerate all data without any environment configuration. Users who wish to reproduce or modify the dataset only need to (i) pull the versioned Docker image corresponding to the release tag, (ii) obtain the released blend-files, and (iii) invoke mkdataset.py with the published configuration.