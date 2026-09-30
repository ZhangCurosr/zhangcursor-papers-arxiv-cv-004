# Pow3R-SLAM: Real-Time RGB-D SLAM with 3D Reconstruction Priors

Christopher Kolios<sup>1</sup>, Ishaan Mehta<sup>1</sup>, Sasa Janjic<sup>2</sup>, Yeganeh Bahoo<sup>1</sup> and Sajad Saeedi<sup>3</sup>

Abstract— We present Pow3R-SLAM, a real-time RGB-D simultaneous localization and mapping (SLAM) system that uses Pow3R for tracking and mapping. Inspired by MASt3R-SLAM, a recent work on monocular SLAM using two-view 3D reconstruction priors, we extend the work to incorporate depth as a prior on the network’s prediction, rather than as geometry to fuse. Where traditional RGB-D SLAM systems struggle with sparsity in the depth images, Pow3R utilizes the available depth to give a better-conditioned pointmap, while inferring the depths in empty regions from the two-view photometric, depth, and intrinsic data. Evaluated against MASt3R-SLAM following its protocol on 24 sequences from TUM, 7-Scenes, and Replica, Pow3R-SLAM runs 1.6× faster in wall time, has 15% lower mean trajectory error, a 3.1× lower unscaled error, and produces denser maps, with a 30% lower Chamfer distance. We also introduce a hybrid variant that runs 2.1× faster than MASt3R-SLAM at 25.3 frames per second (FPS), while maintaining improved tracking and mapping accuracy. Against ORB-SLAM3 in RGB-D mode, Pow3R-SLAM is more accurate on TUM, 7-Scenes, and ETH3D-SLAM, and completes every TUM sequence. While Pow3R-SLAM can struggle on a small set of self-similar scenes, its overall performance shows that adding depth as a prior for two-view 3D reconstruction SLAM can be beneficial. A project webpage is available at: https://ChrisKolios.github.io/Pow3R-SLAM, and code will be made open-source upon acceptance.

## I. INTRODUCTION

Simultaneous localization and mapping (SLAM), the problem of mapping a scene while jointly localizing the input sensor relative to the map, is crucial for autonomous robots, augmented reality systems, and self-driving cars, all of which must map and track unknown environments. However, visual SLAM, which relies solely on cameras as input, can fail on visually similar but geometrically distinct scenes, and cannot observe metric scale without a reference. RGB-D SLAM adds a depth sensor so the map and pose can be optimized against both photometric and geometric information.

Classical depth-based SLAM systems such as KinectFusion [1] and RGB-D systems such as ElasticFusion [2] treat the depth as geometry, fusing it into the map and tracking against the fused surface. However, real-world depth images are not a perfect representation of the true scene geometry. Depth sensors leave holes on absorptive, dark or shiny surfaces, and are inconsistent at object boundaries (Fig. 1 (c, d)). Systems that fuse these depths as geometry inherit all of these defects, but systems that do not incorporate depth discard a key metric measurement.

![](images/41febffa04e19ae722b4e735e8d5d9875ce3c64a1b883ddc6b36aaa32af513f5.jpg)  
Fig. 1. Sensor depth vs. Pow3R output. Given input frames ((a), (b)), with corresponding depth maps ((c), (d)) from the TUM fr1/desk scene [3], (e) shows the pointmap result from the back-projected depthmaps, and (f) shows the pointmap result from passing the input RGB-D into Pow3R’s network. Pow3R maintains flat surfaces, and reconstructs the controller where the Kinect sensor lacks the data to do so (fills holes and regularizes surfaces).

Two-view 3D reconstruction prior works, including DUSt3R [4], MASt3R [5], and Pow3R [6] have shown that vision transformers (ViTs) can learn the relative geometry of two images. When passing in a pair of images, these networks can output a set of pointmaps in a common frame of reference, from which the relative poses of the cameras that took those images can be recovered. In particular, Pow3R demonstrated that passing depth, camera intrinsics, and relative pose (or any subset thereof) can condition the network, offering more accurate and geometrically consistent surfaces than RGB input alone, and inferring depth where none is provided. In Fig. 1, the back-projected depth (e) cannot fill in the controller, but Pow3R’s pointmap (f) can.

Pow3R-SLAM builds on MASt3R-SLAM [7], a monocular system that tracks and maps with MASt3R’s pointmaps, and extends it to a full RGB-D system around Pow3R. Sensor depth enters at three separable sites: conditioning the network’s prediction, fixing the metric scale of every pointmap, and anchoring keyframe geometry inside the global optimization. The map is always the network’s output, so holes are filled and surfaces regularized as in Fig. 1(f).

Pow3R-SLAM adopts MASt3R-SLAM’s projective matching and second-order backend formulation, adapted to new inputs. Since Pow3R has no matching head, loop retrieval and per-correspondence confidence are rebuilt from its encoder tokens and confidence head, and map-only keyframes densify the map. A hybrid variant tracks alternate frames by iterative closest point (ICP) on the sensor depth, halving the network cost.

![](images/ef88369857c0b332b9a1fe406d608419de1be68d08f569c409121862c3583277.jpg)  
Fig. 2. Pow3R-SLAM pipeline. Pow3R-SLAM is an RGB-D SLAM system that uses sensor depth as a prior on a two-view pointmap network. The front end (1-3) runs on every frame. (1) Pow3R predicts pointmaps and confidences for the frame and its keyframe, conditioned on sensor depth and intrinsics (Sec. III-B). (2) Each prediction is made metric against the sensor depth (Eq. 1). (3) The frame is tracked in Sim(3) and may become a keyframe (Secs. III-C–III-D). The backend (4, 5) runs on every new keyframe. (4) Loop closure on Pow3R’s encoder tokens adds edges in familiar locations, and relocalization locates frames the tracker has lost. (5) A global optimization refines every keyframe pose, with keyframe depths anchored to the sensor and confidence-calibrated edges (Sec. III-E.3, Eqs. 5–7). Blue: the sensor-depth path. Orange: optional components, including the hybrid variant, which tracks alternate frames by ICP instead of Pow3R (Sec. III-G), and map-only keyframes (Sec. III-F). ICP frames never become keyframes, so every keyframe comes from a Pow3R pass. Inputs and outputs: TUM fr1/desk.

Our main contributions are:

• To the best of our knowledge, the first RGB-D SLAM system in which sensor depth is a prior on a two-view pointmap network, entering at three separable sites.

• Loop retrieval and per-correspondence confidence rebuilt from Pow3R’s encoder and confidence head, plus map-only keyframes that densify the map, and a hybrid ICP front end mode.

• Evaluation on four benchmarks against MASt3R-SLAM and ORB-SLAM3 on accuracy, reconstruction, runtime, per-site ablations, and sensitivity to depth quality.

## II. RELATED WORK

Monocular SLAM uses RGB data from a single source as input to the system. Feature-based systems such as ORB-SLAM [8] estimate poses from sparse feature correspondences across views. Dense monocular SLAM algorithms such as DTAM [9] use the entire image to create a dense model of the scene. These methods are determined only up to a similarity transform, leaving metric scale unknown.

RGB-D SLAM systems add a depth sensor to the camera, or use depth alone. KinectFusion [1] fuses depth into a truncated signed distance field and tracks against it. ElasticFusion [2] uses both RGB and depth images to jointly minimize photometric and geometric pose errors for tracking. ORB-SLAM2 [10] extends ORB-SLAM’s pipeline to depth, and ORB-SLAM3 [11] adds inertial measurement unit (IMU) data and multi-map support. More modern systems such as DROID-SLAM [12] pair optical flow estimation with bundle adjustment to jointly optimize pose and dense perpixel depth. Neural map representations have recently been popularized, with the map stored as a network or optimizable set of primitives. iMAP [13] and NICE-SLAM [14] optimize implicit fields against depth, and GS-SLAM [15] optimizes a scene of 3D Gaussians, at a cost in frame rate. All of these works consume depth as geometry to fuse, or use it as a supervision target.

A recent class of ViTs called two-view 3D reconstruction priors takes two images and outputs dense aligned 3D point clouds. These common-frame point clouds can be used to recover camera pose (such as in DUSt3R [4]). MASt3R [5], a successor work, added per-pixel features for alignment, which improved pose recovery. Pow3R [6], a further successor, accepts any subset of depth, camera intrinsics, and relative pose, as optional inputs, which further improves the resulting point cloud. Extending this beyond pairs, feed-forward scene models take all views at once, such as VGGT [16], which predicts poses and depth for a whole set of frames.

Recent SLAM with reconstruction priors uses either two-view or scene prior models to track and map. MASt3R-SLAM [7] uses MASt3R for real-time tracking and mapping by optimizing projective matches, with a Sim(3) pose graph and an aggregated selective match kernel (ASMK) [17] retrieval, and is the system we extend in this work. Scene-prior systems include VGGT-SLAM [18], which incrementally aligns VGGT submaps, and SLAM3R [19], which fuses local reconstructions into a global scene. However, to the best of our knowledge, no reconstruction-prior-based SLAM works have investigated how depth might be used as an input.

## III. METHODS

Fig. 2 displays the method overview, highlighting how Pow3R-SLAM works, and where both depth and the ICP hybrid approach play a role in the system.

## A. Preliminaries

Throughout the text, we adopt the conventions of MASt3R-SLAM [7]. Given two views $i , j ,$ Pow3R [6] defines a network $\mathcal { F }$ that takes an image pair $\mathcal { T } ^ { i } , \mathcal { T } ^ { j } \in \mathbb { R } ^ { H \times W \times 3 }$ and outputs pointmaps $\mathbf { X } _ { i } ^ { i } , \mathbf { X } _ { i } ^ { j } , \mathbf { X } _ { j } ^ { j }$ (where $\mathbf { X } _ { i } ^ { j }$ denotes the pointmap of image $\mathcal { T } ^ { j }$ in the coordinate system of the camera from view i) and their associated confidences $\mathbf { C } _ { i } ^ { i } , \mathbf { C } _ { i } ^ { j } , \mathbf { C } _ { j } ^ { j } .$

Pow3R can optionally take a subset of auxiliary information $\Omega \subseteq \{ \mathbf { Z } ^ { i } , \mathbf { Z } ^ { j } , \mathbf { K } ^ { i } , \mathbf { K } ^ { j } , \mathbf { P } _ { i j } \}$ , including depthmaps ${ \bf z } ^ { i } , { \bf Z } ^ { j } \in$ $\mathbb { R } ^ { H \times W \times \overline { { 1 } } }$ (called D in [6]) with their validity masks M<sup>i</sup>, $\mathbf { M } ^ { j } \in$ $\{ 0 , 1 \} ^ { H \times W }$ , which accompany a depth map rather than being a prior of their own and indicate pixels with valid depth, camera intrinsics K<sup>i</sup>, $\mathbf { K } ^ { j } \in \mathbb { R } ^ { 3 \times 3 }$ , and relative pose $\mathbf { P } _ { i j } \in \mathbb { R } ^ { 4 \times 4 }$ We write a forward pass with priors Ω as $\mathcal { F } _ { P } ( \bar { \mathcal { T } } ^ { i } , \bar { \mathcal { T } } ^ { j } , \Omega )$ outputting $( \mathbf { X } _ { i } ^ { i } , \mathbf { C } _ { i } ^ { i } ) , ( \bar { \mathbf { X } _ { i } ^ { j } } , \mathbf { C } _ { i } ^ { j } ) , ( \mathbf { X } _ { j } ^ { j } , \mathbf { C } _ { j } ^ { j } )$ . As Pow3R-SLAM is an RGB-D approach, we assume that $\mathbf { Z } ^ { i } , \mathbf { Z } ^ { j }$ are always available. ${ \bf P } _ { i j }$ is what we solve for and is never supplied, so $\pmb { \Omega } = ( \mathbf { Z } ^ { i } , \mathbf { Z } ^ { j } , \mathbf { K } )$ throughout this work. Pow3R-SLAM’s calibrated mode assumes that $\mathbf { K } ^ { i } = \mathbf { K } ^ { j }$ for all frames (as does MASt3R-SLAM’s), and a fixed calibration is required regardless to register depth to RGB. All results use this mode.

We maintain the same pose formulation as MASt3R-SLAM, with $\mathbf { T } \in \mathbf { S i m } ( 3 )$ , and updates $\tau \in$ sim3 applied by a left-plus operator. $\mathbf { T }  \pmb { \tau } \oplus \mathbf { T } \triangleq \mathrm { E x p } ( \tau ) \circ \mathbf { T }$ , where T holds a rotation $\mathbf { R } \in \mathbf { S 0 } ( 3 )$ , a translation $\mathbf { t } \in \mathbb { R } ^ { 3 }$ , and a scale $s > 0$

Pow3R-SLAM uses a known pinhole camera as its primary setting. Images with lens distortion are undistorted first. K is adjusted when the image is resized for Pow3R (long side at 512 pixels) and this resized K is used throughout the pipeline. Let $\mathcal { R } _ { \mathbf { K } } ( \mathbf { X } ) [ \mathbf { p } ] \triangleq z ( \mathbf { X } [ \mathbf { p } ] ) \mathbf { K } ^ { - 1 } [ u , \nu , 1 ] ^ { \top }$ be the operation that keeps a pointmap’s depth $z ( \cdot )$ at pixel $\mathbf { p } =$ $\left( u , \nu \right)$ and returns the point to its calibrated ray. Both selfview pointmaps are passed through $\mathcal { R } _ { \mathbf { K } }$ before use, so only the network’s depths reach the estimator.

## B. Known Depth and Calibration

A primary focus of this work is the integration of depth data into the pipeline. Sensor depth is involved in three areas of Pow3R-SLAM (Fig. 2), denoted: cond, scale, and anchor.

1) cond: (conditioning) feeds depth in as an optional input to Pow3R. Following Pow3R’s convention [6], each depth map is normalized by its mean over valid pixels, $\mathbf { Z } / \bar { z }$ with $\bar { z } = \arg \mathbf { \mathbf { \mathbf { z } _ { M } ( Z ) } }$ , together with its validity mask $\mathbf { M } = [ \mathbf { Z } >$ 0]. The intrinsics are not passed as a matrix, and are instead expanded into a dense ray image, evaluating $\mathbf { K } ^ { - 1 } [ u , \nu , 1 ] ^ { \top }$ at every pixel $\left( u , \nu \right)$ . Both maps are then cut into the same $1 6 \times 1 6$ patch grid as the image, and embedded into one token per patch. Those tokens are injected inside the network’s first encoder block, where a learned projection of each is added to the token of the image patch it covers. Pow3R therefore receives $\mathcal { F } _ { P } ( \mathcal { T } ^ { i } , \mathcal { T } ^ { j } , \Omega )$ with $\pmb { \Omega } = ( \mathbf { Z } ^ { i } , \mathbf { Z } ^ { j } , \mathbf { K } )$

2) scale: restores metric scale from the (metric) depth images. In each forward pass, one scalar is estimated from the self-view pointmap $\mathbf { X } = \mathbf { X } _ { i } ^ { i }$ against its sensor depth $\mathbf { Z } =$ $\mathbf { Z } ^ { i } .$ , and applied to both outputs of that pass $( \mathbf { X } _ { i } ^ { i }$ and $\mathbf { X } _ { i } ^ { j } ,$ which share a single gauge, i.e. the same unknown scale and coordinate frame). Let the per-pixel depth ratio $\begin{array} { r } { r _ { \mathbf { p } } = \frac { \mathbf { Z } [ \mathbf { p } ] } { z ( \mathbf { X } | \mathbf { p } | ) } . } \end{array}$ over valid pixels p with $\mathbf { M } [ \mathbf { p } ] = 1$ and $z ( \mathbf { X } [ \mathbf { p } ] ) > 0$ (such that $z ( \mathbf { x } )$ is the depth of a point $\mathbf { x } \in \mathbf { X } )$ ). Then, given median $( r _ { \mathbf { p } } ) =$ r¯, our scaling formula is defined as:

$$
s ( \mathbf { X } , \mathbf { Z } ) \triangleq \exp ( \mathrm { a v g } _ { \mathbf { p } } \big ( \mathrm { l o g } \big ( \mathrm { c l i p } ( r _ { \mathbf { p } } , 0 . 1 \cdot \bar { r } , 1 0 \cdot \bar { r } ) \big ) \big ) \big ) ,\tag{1}
$$

where $\cdot \operatorname* { l i p } ( a , b , c ) = \operatorname* { m i n } ( \operatorname* { m a x } ( a , b ) , c )$ , applied when at least 50 pixels are valid (a floor against near-empty depth maps below which the pass is left unscaled). The geometric mean is the minimizer of the squared log-depth residuals that our solvers use (the log-depth row of Eq. 3, and Eq. 6), and the clip rejects outliers. The scale is the geometric mean of the per-pixel depth ratios, with ratios more than $1 0 \times$ from their median pulled back to that bound.

3) anchor: brings the sensor into the backend. Prior to each backend solve, the depth of each keyframe pointmap is replaced along its known rays by the sensor’s depth wherever the sensor has data. The substituted depths are not clamped in the solution, acting instead as a soft constraint in the backend optimization (Sec. III-E.3).

## C. Pointmap Matching and Fusion

We adopt the projective matching approach of MASt3R-SLAM. For a function $\psi ( \mathbf { x } )$ that normalizes a point to a unit ray, the match for $\mathbf { x } \in \mathbf { X } _ { i } ^ { j } \mathrm { i s } .$ , following [7], the pixel $\begin{array} { r } { \mathbf { p } ^ { * } = \arg \operatorname* { m i n } _ { \mathbf { p } } \| \psi ( [ \mathbf { X } _ { i } ^ { i } ] _ { \mathbf { p } } ) - \psi ( \dot { \mathbf { x } } ) \| ^ { 2 } } \end{array}$ , solved by a per-point Levenberg–Marquardt with ten iterations. For new keyframes we initialize from an identity mapping, we employ nextframe initialization for tracking, accept a match when its ray residual converges below $1 0 ^ { - 6 }$ , and reject outliers at $> 0 . 1$ m in 3D space. The only difference is that we do not use perpixel features to further refine the matches, as Pow3R has no such features.

After each tracked frame (Sec. III-D), we use the same criteria as MASt3R-SLAM to add new keyframes. Let $\omega _ { k }$ be the smaller of the fraction of keyframe pixels matched and the fraction of distinct keyframe pixels those matches land on. A keyframe is added when $\omega _ { k } < 0 . 3 3 3$ . Once pose is solved, we update the keyframe’s geometry via $\mathbf { X } _ { k } ^ { k } \gets$ $\mathbf { T } _ { k f } \mathbf { X } _ { f } ^ { k }$ (where $\mathbf { X } _ { k } ^ { k }$ is the keyframe pointmap, $\mathbf { T } _ { k f }$ is the relative pose between the incoming frame and the keyframe (Sec. III-D), and $\mathbf { X } _ { f } ^ { k }$ the keyframe’s pointmap in the incoming frame’s coordinates). While MASt3R-SLAM uses a running weighted average filter, for each pixel we instead keep the most confident prediction the keyframe has received:

$$
( \tilde { \mathbf { X } } _ { k } ^ { k } [ \mathbf { p } ] , \tilde { \mathbf { C } } _ { k } ^ { k } [ \mathbf { p } ] ) \gets ( \mathbf { X } _ { k } ^ { k } [ \mathbf { p } ] , \mathbf { C } _ { f } ^ { k } [ \mathbf { p } ] ) \mathrm { ~ i f ~ } \mathbf { C } _ { f } ^ { k } [ \mathbf { p } ] > \tilde { \mathbf { C } } _ { k } ^ { k } [ \mathbf { p } ] .\tag{2}
$$

Since each pass has its own scalar (from Eq. 1), averaging predictions would average scale error into the keyframe.

## D. Tracking

Tracking estimates the pose of each incoming frame $f$ relative to the current keyframe k from a single pass $\mathcal { F } _ { P } ( \mathcal { T } ^ { f } , \mathcal { T } ^ { k } , \Omega )$ per frame, following MASt3R-SLAM’s frame-to-keyframe scheme. For an incoming point transported into the keyframe camera $\mathbf { X } _ { m } .$ , and the keyframe’s own canonical point at the matched pixel ${ \bf y } _ { n } .$ , let $\mathbf { X } _ { m } =$ $\mathbf { T } _ { k f } \mathcal { R } _ { \mathbf { K } } ( \mathbf { X } _ { f } ^ { f } ) _ { m }$ and $\mathbf { y } _ { n } = \mathcal { R } _ { \mathbf { K } } ( \tilde { \mathbf { X } } _ { k } ^ { k } ) _ { n }$ (Sec. III-A).

With Π<sub>K</sub> the projection under K, a correspondence $( m , n )$ in the match set ${ \bf { m } } _ { f , k }$ gives:

$$
\mathbf { r } _ { m n } = \left[ \begin{array} { c } { \pi _ { \mathbf { K } } ( \mathbf { y } _ { n } ) - \pi _ { \mathbf { K } } ( \mathbf { x } _ { m } ) } \\ { \log z ( \mathbf { y } _ { n } ) - \log z ( \mathbf { x } _ { m } ) } \end{array} \right] .\tag{3}
$$

Following the weighting of Eq. (5) in [7], a residual with confidence q carries the covariance $w ( q , \sigma ^ { 2 } ) = \sigma ^ { 2 } / q$ when $q > q _ { \mathrm { m i n } }$ and is discarded $( w = \infty )$ otherwise. $\| \mathbf { r } \| _ { \rho , w }$ denotes the Huber norm (threshold 1.345) of the whitened residual $\mathbf { r } / { \sqrt { w } } .$ . In tracking, Pow3R provides no per-match confidence, so $q \equiv 1$ , and outliers are rejected by the matching tests of Sec. III-C (the ray-residual convergence check and the 0.1 m 3D rejection) rather than by the confidence gate.

Tracking minimizes the reprojection error:

$$
E _ { T I } = \sum _ { ( m , n ) \in \mathbf { m } _ { f , k } } \| \varPi _ { \mathbf { K } } ( \mathbf { y } _ { n } ) - \varPi _ { \mathbf { K } } ( \mathbf { x } _ { m } ) \| _ { \rho , w ( 1 , \sigma _ { \mathrm { p x } } ^ { 2 } ) } ,\tag{4}
$$

with $\sigma _ { \mathrm { p x } } = 1$ pixel. We use Gauss–Newton for at most 50 iterations to optimize reprojection error and recover the relative pose $\mathbf { T } _ { k f }$ . As in MASt3R-SLAM, a second, lightly weighted term enters the same solve (the log-depth row of Eq. 3). This prevents degeneracy under pure rotation, where reprojection leaves depth unobserved, and it alone observes the scale of $\mathbf { T } _ { k f } \in \mathbf { S i m } ( 3 )$ , since a similarity scaling leaves Π<sub>K</sub> invariant while log z shifts by log s. We use a loose $\sigma _ { z } ^ { t } = 1 0$ (superscripts t and $g$ denoting tracking and global solves), admitting a small percent of per-frame scale noise for the backend to reconcile. If fewer than 5% of pixels survive the matching, or the normal equations fail to factorize, the frame is deemed lost and relocalization (Sec. III-E.2) is triggered. Upon adding a new keyframe, a bidirectional edge is added to the previous keyframe.

## E. Loop Closure, Relocalization, and Backend

1) Loop Closure: recognizes re-visited areas and corrects the relative keyframe poses. We use the same ASMK [17], [20] retrieval as MASt3R-SLAM, but in place of MASt3R’s encoder tokens we use an unconditioned Pow3R encoder pass over the keyframe image, computed once per keyframe. We refit the training-free ASMK head to Pow3R’s tokens, reestimating the principal component analysis (PCA) whitening and reducing the k-means codebook from 65536 to 16384 centroids. New keyframes query the top 3 candidates above a score threshold of 0.005. The candidates are each decoded by a Pow3R pass in each direction and matched projectively as in Sec. III-C, with a loop edge added when the minimum match fraction is at least $\omega _ { l } = 0 . 2 5$ (the edge to the preceding keyframe is unconditionally added).

2) Relocalization: occurs when tracking is lost (Sec. III-D). We follow the same approach as MASt3R-SLAM here, querying the retrieval database with the same score threshold but a stricter acceptance rule. The retrieved frame is added as a keyframe only if every retrieved candidate clears a matchfraction floor of 0.3, after which its pose is initialized from the first candidate and a backend solve is run.

3) Backend: We reuse the second-order optimization scheme of MASt3R-SLAM, bidirectionally minimizing the calibrated residual over all edges via Gauss–Newton with a sparse Cholesky solve, with the first keyframe’s pose held fixed, and at most ten iterations per new keyframe. There are two primary differences in our method.

First, prior to each solve, an anchor step replaces (only for the backend solve) the depth for each keyframe pointmap with its sensor values along the rays that form the pointmap, where the sensor has data $( \hat { \mathbf { X } } _ { i } ^ { i } = \mathcal { R } _ { \mathbf { K } } ( \tilde { \mathbf { X } } _ { i } ^ { i } )$ with $z  \mathbf { Z } ^ { i }$ on M<sup>i</sup>).

As the sensor is typically more accurate where it has data, we set $\sigma _ { z } ^ { g } = 0 .$ 1 for the log-depth residual here (against $\sigma _ { z } ^ { t } =$ 10 in tracking).

For each edge $( i , j )$ in the set of all edges $\mathcal { E } ,$ traversed in both directions $\mathcal { E } ^ { \pm }$ , with $\mathbf { T } _ { i j } = \mathbf { T } _ { W C _ { i } } ^ { - 1 } \mathbf { T } _ { W C _ { j } } , \hat { \mathbf { x } } _ { m } =$ $\mathbf { T } _ { i j } \mathcal { R } _ { \mathbf { K } } \big ( \hat { \mathbf { X } } _ { j } ^ { j } \big ) _ { m } .$ and $\hat { \mathbf { y } } _ { n } = \mathcal { R } _ { \mathbf { K } } \big ( \hat { \mathbf { X } } _ { i } ^ { i } \big ) _ { n }$ , the backend minimizes the sum ${ \check { E } } _ { \Pi } ^ { g } + E _ { z } ^ { g }$ of:

$$
E _ { \varPi } ^ { g } = \sum _ { ( i , j ) \in \mathcal { E } ^ { \pm } } \sum _ { ( m , n ) \in { \bf m } _ { i j } } \Vert \varPi _ { \bf K } ( \hat { \bf y } _ { n } ) - \varPi _ { \bf K } ( \hat { \bf x } _ { m } ) \Vert _ { \rho , w ( w _ { m n } , \sigma _ { \mathrm { p x } } ^ { 2 } ) } ,\tag{5}
$$

$$
E _ { z } ^ { g } = \sum _ { ( i , j ) \in \mathcal { E } ^ { \pm } } \sum _ { ( m , n ) \in { \bf m } _ { i j } } \| \log z ( \hat { \bf y } _ { n } ) - \log z ( \hat { \bf x } _ { m } ) \| _ { \rho , w ( w _ { m n } , ( \sigma _ { z } ^ { g } ) ^ { 2 } ) } ,\tag{6}
$$

with hats indicating an anchored pointmap, $\sigma _ { \mathrm { p i x } } = 1$ px as in tracking, and $w _ { m n } \ ( \mathrm { E q . \ 7 } )$ filling the role of $q _ { m n } .$

Anchoring is disabled when scale is disabled, and the anchored copy never leaves the backend.

Second, as we do not have MASt3R’s matching confidence, we rebuild the per-correspondence weight from Pow3R’s confidences as:

$$
w _ { m n } = { \bf Q } _ { m n } \tilde { \bf C } _ { i , m } ^ { i } \tilde { \bf C } _ { j , n } ^ { j } , ~ \mathrm { w h e r e } ~ { \bf Q } _ { m n } = \phi ( \sqrt { { \bf C } _ { i , m } ^ { i } { \bf C } _ { i , n } ^ { j } } ) ,\tag{7}
$$

where $\phi$ is a piecewise-linear quantile map with 1024 knots, fitted once so that its output distribution matches MASt3R-SLAM’s matching confidence over 109630 correspondences from eight development sequences (Sec. IV). $\mathbf { Q } _ { m n }$ reproduces MASt3R-SLAM’s gate $\mathbf { Q } _ { m n } > 1 . 5$ without threshold re-tuning or retraining. The two C<sup>˜</sup> factors downweight low-confidence geometry, and in practice the $\mathbf { Q } _ { m n } >$ 1.5 gate carries most of the effect of $w _ { m n }$

## F. Map-Only Keyframes

The keyframe rate that is ideal for pose estimation leads to maps that are too sparse, but raising it changes the estimator, so we look to decouple map density from the estimator. For this, we add map-only keyframes, which carry geometry but no estimator state. A network-tracked frame with $0 . 3 3 3 \leq$ $\omega _ { k } < 0 . 4 5$ is stored when at least 5 input frames have passed since the previous addition, up to 200 per sequence. For each map-only keyframe $f ,$ we store its rescaled pointmap $\mathbf { X } _ { f } ^ { f } ,$ confidence ${ \bf C } _ { f } ^ { f }$ , image $\mathcal { T } ^ { f } .$ , and its pose relative to its keyframe $\mathbf { T } _ { k f }$

At export, the world pose is recomposed against the keyframe’s final pose, $\mathbf { T } _ { W C _ { f } } = \mathbf { T } _ { W C _ { k } } \mathbf { T } _ { k f }$ in Sim(3), so backend corrections and loop closures propagate to them. The final exported map is the union of the keyframe and maponly pointmaps under a single confidence threshold. This densifies the map, without modifying trajectories.

## G. Hybrid ICP Variant

In both MASt3R- and Pow3R-SLAM, the network forward pass dominates the tracker’s cost, which greatly affects overall runtime. Inspired by MASt3R-SLAM’s success when simulating real-time performance by skipping every 2nd frame and existing depth-based approaches, our hybrid variant processes every 2nd frame via point-to-plane ICP [21], [22] on the depth sensor input, replacing the network call for those frames. Let $\mathbf { x _ { p } } = \mathbf { Z } ^ { f } [ \mathbf { p } ] \mathbf { K } ^ { - 1 } [ u , \nu , \bar { 1 } ] ^ { \top }$ be frame $f ^ { \ast } \mathbf { s }$ sensor depth back-projected along its rays and ${ \bf q _ { p } } = \Pi _ { \bf K } ( { \bf T } _ { k f } { \bf x _ { p } } )$ its projective association (nearest pixel) into the keyframe, recomputed at every ICP iteration. The sum runs over the set P of pixels at which both the sensor depth and the keyframe pointmap carry a valid central-difference normal (excluding borders, hole neighbours, and depth steps above 5%). With $\mathbf { n _ { q _ { p } } }$ the keyframe’s normal at ${ \bf q _ { p } } ,$ , the point-to-plane error is minimized against the keyframe’s canonical pointmap [1], [21]:

$$
E _ { \mathrm { i c p } } = \sum _ { \mathbf { p } \in \mathbf { P } } \| \big ( \mathbf { n } _ { \mathbf { q } _ { \mathbf { p } } } ^ { \top } \big ( \mathbf { T } _ { k f } \mathbf { x } _ { \mathbf { p } } - \tilde { \mathbf { X } } _ { k } ^ { k } [ \mathbf { q } _ { \mathbf { p } } ] \big ) \big ) \| _ { \rho ^ { \prime } } ,\tag{8}
$$

where $\rho ^ { \prime }$ is a Huber kernel at 2 cm (a metric distance, unlike $\rho )$ . Our solve is coarse-to-fine over 3 levels with iteration counts (6,4,3) on $\mathbf { S E } ( 3 )$ since the frame inherits its keyframe’s scale, gated on correspondence distance (10cm) and normal rotation (30<sup>◦</sup>), and falls back to network tracking whenever the pose is not finite, the finest-level cost rises by > 5%, or < 30% of the associable pixels are inliers.

ICP-tracked frames never become keyframes, reach the backend, or add points to the map.

## IV. RESULTS

We evaluate localization on TUM RGB-D [3], 7- Scenes [23], Replica (synthetic) [24], and ETH3D-SLAM [25], and geometry on 7-Scenes and Replica, which have ground-truth geometry.

7-Scenes provides its Kinect colour and depth images unregistered, with only the depth camera’s intrinsics, which misplaces the edges between depth and colour pixels by 20–40 px. For every system that uses depth, we therefore resample each colour image into the depth camera through the depth image. To calibrate, we take the colour focal length and radial distortion (from COLMAP [26] self-calibration of the colour images) and the 2.7 cm baseline and principal point from depth-colour edge alignment. One calibration is shared by all scenes, and neither step uses ground truth or trajectory error. The fitting corpus of φ (Sec. III-E.3) overlaps the evaluation panel on four sequences (fr1/teddy, room0, pumpkin, stairs). As φ only preserves MASt3R-SLAM’s threshold meaning, and refitting it without the four moves them by at most 18 mm (the largest, stairs, an improvement), this is not a material leak. The retrieval codebook was fitted on five TUM fr2/fr3 and eight ETH3D-SLAM training sequences, which we exclude from evaluation.

All runs use a Ryzen 5900X 3.7 GHz CPU and an NVIDIA GeForce RTX 4090. As in MASt3R-SLAM’s protocol, every

TABLE I  
ATE ROOT-MEAN SQUARE ERROR (RMSE) (M) ON TUM RGB-D FR1 AND 7-SCENES. BOLD, UNDERLINE: BEST, SECOND BEST. †: CALIBRATED MONOCULAR RESULT AS REPORTED IN [7]. FRESH: OUR RE-RUN. X: LOST TRACK, WITH MEANS OVER COMPLETED SCENES. ∗: UNSCALED SE(3) ALIGNMENT.
<table><tr><td colspan="2"></td><td colspan="2">ORB-SLAM3</td><td></td><td colspan="3">Pow3R-SLAM</td></tr><tr><td>scene</td><td>DROID†↓</td><td>Sim3↓</td><td> $\mathbf { S E 3 } ^ { * } \downarrow$ </td><td>MASt3R fresh↓</td><td>Sim3↓</td><td>SE3*↓</td><td>hyb. Sim3↓</td></tr><tr><td colspan="8">TUM RGB-D fr1</td></tr><tr><td>360</td><td>0.111</td><td>0.134</td><td>0.228</td><td>0.049</td><td>0.040</td><td>0.048</td><td>0.042</td></tr><tr><td>desk</td><td>0.018</td><td>0.018</td><td>0.018</td><td>0.016</td><td>0.019</td><td>0.035</td><td>0.017</td></tr><tr><td>desk2</td><td>0.042</td><td>X</td><td>X</td><td>0.024</td><td>0.024</td><td>0.093</td><td>0.022</td></tr><tr><td>floor</td><td>0.021</td><td>X</td><td>X</td><td>0.025</td><td>0.021</td><td>0.025</td><td>0.025</td></tr><tr><td>plant</td><td>0.016</td><td>0.022</td><td>0.026</td><td>0.020</td><td>0.019</td><td>0.020</td><td>0.016</td></tr><tr><td>room</td><td>0.049</td><td>0.073</td><td>0.084</td><td>0.061</td><td>0.042</td><td>0.044</td><td>0.046</td></tr><tr><td>rpy</td><td>0.026</td><td>0.035</td><td>0.040</td><td>0.027</td><td>0.019</td><td>0.023</td><td>0.017</td></tr><tr><td>teddy</td><td>0.048</td><td>X</td><td>X</td><td>0.041</td><td>0.049</td><td>0.066</td><td>0.032</td></tr><tr><td>xyz</td><td>0.012</td><td>0.012</td><td>0.013</td><td>0.009</td><td>0.006</td><td>0.007</td><td>0.005</td></tr><tr><td>mean</td><td>0.038</td><td>0.049 (6)</td><td>0.068 (6)</td><td>0.030</td><td>0.027</td><td>0.040</td><td>0.025</td></tr><tr><td colspan="8"></td></tr><tr><td>7-Scenes chess</td><td>0.036</td><td>0.033</td><td>0.038</td><td>0.038</td><td>0.027</td><td>0.035</td><td>0.027</td></tr><tr><td>fre</td><td>0.027</td><td>0.023</td><td>0.028</td><td>0.029</td><td>0.025</td><td>0.031</td><td>0.024</td></tr><tr><td>heads</td><td>0.025</td><td>0.014</td><td>0.017</td><td>0.017</td><td>0.012</td><td>0.015</td><td>0.014</td></tr><tr><td>office</td><td>0.066</td><td>0.096</td><td>0.098</td><td>0.092</td><td>0.085</td><td>0.085</td><td>0.086</td></tr><tr><td>pumpkin</td><td>0.127</td><td>0.127</td><td>0.127</td><td>0.084</td><td>0.077</td><td>0.087</td><td>0.083</td></tr><tr><td>kitchen</td><td>0.040</td><td>0.047</td><td>0.048</td><td>0.060</td><td>0.038</td><td>0.038</td><td>0.039</td></tr><tr><td>stairs</td><td>0.026</td><td>0.045</td><td>0.049</td><td>0.016</td><td>0.044</td><td>0.051</td><td>0.062</td></tr><tr><td>mean</td><td>0.049</td><td>0.055</td><td>0.058</td><td>0.048</td><td>0.044</td><td>0.049</td><td>0.048</td></tr></table>

2nd frame is processed. Pow3R-SLAM and MASt3R-SLAM accuracy is deterministic, and every runtime is the mean of two exclusive repeats, with the half-range reported in Table II. All results use the single-threaded mode, in which the front end waits for the backend to drain. Using either system’s non-blocking mode changes wall time by under 5%.

## A. Pose Estimation

Following the convention of MASt3R-SLAM, we report the root-mean-square-error (RMSE) of the absolute trajectory error (ATE) in meters (Table I). Pose estimation results are from the Sim(3)-aligned keyframe trajectory. We re-run MASt3R-SLAM for direct comparison, which reproduces the published values of [7] to the reported precision on all 16 sequences from TUM and 7-Scenes on native images. On 7- Scenes, all tables report it on the registered images. We also re-run ORB-SLAM3 [11] in RGB-D mode with calibration, at a median of five runs unless otherwise noted (v1.0, loop closure on, stock ORB parameters). In our tables the best result is bolded, the second best underlined, and arrows give the better direction.

1) TUM RGB-D: Pow3R-SLAM has the lowest mean trajectory error on the TUM sequences (Table I), in both hybrid and regular modes. ORB-SLAM3 loses track on $3 / 9$ sequences (desk2, floor, teddy), while all other systems complete all 9. Pow3R-SLAM improves on MASt3R-SLAM on $6 / 9$ sequences, and on the mean (0.027 vs. 0.030 m), with the largest relative gains on room and, in hybrid mode, teddy.

2) 7-Scenes: With colour registered to depth, Pow3R-SLAM has the lowest mean on 7-Scenes (0.044 m, against 0.048 m for MASt3R-SLAM and 0.055 m for ORB-SLAM3), better than MASt3R-SLAM on $6 / 7$ sequences. stairs is the exception (0.044 vs. 0.016 m), which we analyze in Sec. V and Fig. 4. The registration lowered the 7-Scenes mean absolute trajectory error (ATE) of Pow3R-SLAM from 0.065 to 0.044 m and of ORB-SLAM3 from 0.059 to 0.055 m. MASt3R-SLAM receives the same registered images, although it uses no depth. On native images, its 7-Scenes ATE was 0.047 m, which became 0.048 m.

3) Replica: On Replica, Pow3R-SLAM significantly outperforms MASt3R-SLAM (Table II). Mean ATE is almost half that of MASt3R-SLAM (0.009 vs. 0.015 m), with tighter trajectories on 5/8 sequences. Replica’s rendered depth is complete and noise-free, which strengthens conditioning and scale. ORB-SLAM3 is the most accurate system here (mean 0.005 m), for which the classical tracking benefits from exact, hole-free depth, and richly textured renders.

4) ETH3D-SLAM: Due to ETH3D-SLAM’s fast camera motion, we use all frames, as MASt3R-SLAM does. From its 61 training sequences we evaluate a 16-sequence panel which was fixed before any runs. Table II reports the mean over completed sequences, the median, and the area under the success-versus-threshold curve (AUC) for thresholds up to 0.5 m (max. 8 on this panel, which is not comparable to the 61-sequence AUC of [7]). While MASt3R-SLAM completes all 16, and Pow3R-SLAM does not complete 2 (planar 3 spans 5× the ground truth extent and repetitive does not finish), Pow3R-SLAM’s median is lower (0.0135 vs. 0.0280 m) and it wins 12/14 sequences that both complete. Its mean over those 14 (0.059 m) is inflated by sofa 2 at 0.66 m. A sequence counts as completed, for every system, when all keyframe poses are finite and the trajectory span is within 2× the ground-truth span. In comparison, ORB-SLAM3 fails 6 scenes in ETH3D-SLAM, and of the 9 sequences all three algorithms complete its ATE lies between Pow3R-SLAM and MASt3R-SLAM (0.026 vs. 0.014 and 0.038 m, respectively). Counting all scenes, its AUC is 4.76, against 6.33 for Pow3R-SLAM and 7.36 for MASt3R-SLAM, which completes every sequence. The hybrid mode completes the same 14 scenes at a lower mean than MASt3R-SLAM (0.034 vs. 0.039 m on those 14, better on 12/14).

5) Per-frame and Unscaled Scoring: Over the 24-scene panel, the keyframe ATE is a like-for-like comparison between MASt3R- and Pow3R-SLAM, which insert 537 and 525 keyframes, respectively (per-sequence counts differ by up to 8), with hybrid inserting 503. Scored on every frame under rigid SE(3) with no scale fitted, Pow3R-SLAM’s ATE is 0.037 m against MASt3R-SLAM’s 0.116 m, which is 3.1× lower. As MASt3R-SLAM has no metric scale, this gap measures the value of the scale site rather than of the prediction prior (Table IV, L3 vs. L4). Scoring all frames flips only the hybrid’s Sim(3), where its mean rises from 0.026 to 0.035 m against MASt3R-SLAM’s 0.030 to 0.034 m. Under SE(3) the hybrid stays 2.9× ahead, and per-frame loss is likely because ICP tracks worse with sparse depth.

TABLE II  
REPLICA, ETH3D-SLAM AND RUNTIME. KEYFRAME ATE RMSE (M), SIM(3), WITH THE UNSCALED SE(3) MEAN. ETH3D-SLAM: OVER EACH SYSTEM’S COMPLETED SEQUENCES (COUNT IN BRACKETS). RUNTIME: WALL TIME SUMMED OVER THE 24-SCENE PANEL (TUM, 7-SCENES, REPLICA), AT FULL AND SUBSAMPLE-2 (S2). MEAN OF 2 EXCLUSIVE REPEATS PER SYSTEM (± HALF-RANGE), WITH MAST3R-SLAM TIMED ON NATIVE IMAGES. FPS: FRAMES PER SECOND AT S2 OF LOOP TIME EXCLUDING MODEL LOAD.
<table><tr><td>scene / metric</td><td>MASt3R fresh</td><td>Pow3R- SLAM</td><td>Pow3R- SLAM hyb.</td></tr><tr><td></td><td>0.013</td><td>0.009</td><td>0.008</td></tr><tr><td>office0 office1</td><td>0.007</td><td>0.009</td><td>0.013</td></tr><tr><td>office2</td><td>0.022</td><td>0.007</td><td>0.007</td></tr><tr><td>office3</td><td>0.027</td><td>0.010</td><td>0.012</td></tr><tr><td>office4</td><td>0.021</td><td>0.009</td><td>0.011</td></tr><tr><td>room0</td><td>0.006</td><td>0.007</td><td>0.007</td></tr><tr><td>room1</td><td>0.016</td><td>0.007</td><td>0.006</td></tr><tr><td>room2</td><td>0.011</td><td>0.013</td><td>0.013</td></tr><tr><td>mean, Sim(3) ↓</td><td>0.0153</td><td>0.0087</td><td>0.0096</td></tr><tr><td>mean, SE(3) ↓</td><td>0.0930</td><td>0.0107</td><td>0.0116</td></tr><tr><td>ETH3D mean↓</td><td>0.0400 (16)</td><td>0.0588 (14)</td><td>0.0343 (14)</td></tr><tr><td>ETH3D median ↓</td><td>0.0280</td><td>0.0135</td><td>0.0104</td></tr><tr><td>ETH3D AUC↑</td><td>7.3620</td><td>6.3345</td><td>6.5190</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Σ wall s2 (s) ↓</td><td> $1 4 4 3 . 3 \pm 1 4 . 6$ </td><td> $9 0 2 . 6 \pm 4 . 7$ </td><td> ${ \bf 6 9 7 . 6 \pm 1 . 0 }$ </td></tr><tr><td>Σ wall full (s) ↓</td><td> $2 3 9 5 . 4 \pm 0 . 6$ </td><td> ${ \underline { { 1 5 5 5 . 1 \pm 9 . 8 } } }$ </td><td>1146.5 ± 2.0</td></tr><tr><td>FPS mean (s2)↑</td><td>14.0</td><td>18.6</td><td>25.3</td></tr></table>

## B. Geometry Evaluation

We evaluate the exported maps against reference clouds back-projected from every 20th ground-truth depth frame (culled at 4 m, 8 m on Replica), after Sim(3) alignment of the keyframe trajectory, with the estimate thresholded at ${ \bf C } > 1 . 5$ and both clouds subsampled to 200k points. We report accuracy (mean estimate-to-reference distance), completion (the reverse), Chamfer (their average), as well as Chamfer RMSE. As noted in the supplementary materials of MASt3R-SLAM [7], mean Chamfer does not significantly penalize incorrect points, and as our maps cover more of the scene thanks to the map-only keyframes (11.9 M vs. 2.7 M points before subsampling, over the 15 map-scored scenes), RMSE Chamfer provides a fairer comparison. Table III gives the results over 7-Scenes and Replica between MASt3R- and Pow3R-SLAM. Pow3R-SLAM has better quality mapping versus MASt3R-SLAM on both 7-Scenes and (especially) Replica, with Chamfer RMSEs of 0.058 against 0.085 on 7-Scenes, and 0.027 against 0.056 on Replica, and both accuracy and completion improve on both datasets. Over the 15 scenes, mean Chamfer is 3.1 against 4.3 cm (30% lower), and RMSE Chamfer 4.1 against 6.9 cm (40% lower). Providing 7-Scenes’ registered images to MASt3R-SLAM improved its 7-Scenes mean Chamfer from 6.7 cm, to 5.5 cm. Fig. 3 shows three scenes qualitatively.

## C. Runtime

We also analyze runtime (bottom rows of Table II). Compared to MASt3R-SLAM, which on the cross-dataset panel of all 24 scenes finishes in 1443 seconds with a mean FPS of 14.0, Pow3R-SLAM in its regular mode takes 903 seconds (1.6× faster), averaging 18.6 FPS, and its hybrid mode takes only 698 seconds (2.1× faster), averaging 25.3 FPS. At the full frame rate (full) the ratios are 1.5× and 2.1×. The CPUonly ORB-SLAM3 (a sparse tracker with no network pass) takes 371 seconds, running at 51 FPS, for reference. On 7- Scenes, Pow3R-SLAM times include the colour registration. Pow3R has no matching head, so MASt3R-SLAM’s perframe descriptor refinement is absent. The hybrid variant also skips the network pass on every 2nd frame, which occupies 2/3 of the tracker’s time. Pow3R also has fewer parameters than MASt3R (556 M vs. 695 M), yet its network pass takes about as long (41 vs. 44 ms per frame).

(b) MASt3R-SLAM  
(c) Pow3R-SLAM  
(d) top view: GT / MASt3R / Pow3R  
![](images/1e9066509f3fb5ca87fac9c7d876b67b8c7df5d042117eae3cfadc34b6236cc3.jpg)  
Fig. 3. Qualitative reconstruction. Rows: Replica office3, 7-Scenes chess, ETH3D-SLAM mannequin 5 (full frame rate). Columns: (a) reference cloud from ground-truth depth, (b) MASt3R-SLAM and (c) Pow3R-SLAM maps after alignment, coloured by distance to the reference (capped at 10 cm), and (d) top view of the per-frame trajectories (ground truth (GT) black, MASt3R-SLAM orange, Pow3R-SLAM blue).

![](images/d7230f0243658a9fc38686e1de86c5f0cc93f81fe9b3177c6f93e01d36c59c26.jpg)  
Fig. 4. The failure case. 7-Scenes stairs, with the same columns as Fig. 3, (d) GT / M(ASt3R) / P(ow3R). MASt3R- vs. Pow3R-SLAM: 1.5 vs. 4.3 M points, keyframe ATE 0.016 vs. 0.044 m, accuracy 3.2 vs. 6.7 cm, completion 10.5 vs. 5.9 cm. Pow3R-SLAM’s map is more complete but less accurate, which we attribute to the self-similar treads (Sec. V).

TABLE III  
RECONSTRUCTION ON 7-SCENES AND REPLICA. ATE FOR REFERENCE. MEAN ACCURACY (ACC.), COMPLETION (COMP.), CHAMFER (CHAM.), AND RMSE CHAMFER, IN M.
<table><tr><td>dataset</td><td>system</td><td>ATE↓</td><td>Acc.↓</td><td>Comp. ↓</td><td>Cham.↓</td><td>Ch. RMSE↓</td></tr><tr><td></td><td>7-Scenes MASt3R-SLAM fresh</td><td>0.048</td><td>0.051</td><td>0.058</td><td>0.055</td><td>0.085</td></tr><tr><td></td><td>Pow3R-SLAM</td><td>0.044</td><td>0.048</td><td>0.038</td><td>0.043</td><td>0.058</td></tr><tr><td></td><td>Pow3R-SLAM hyb.</td><td>0.048</td><td>0.050</td><td>0.038</td><td>0.044</td><td>0.058</td></tr><tr><td>Replica</td><td>MASt3R-SLAM fresh</td><td>0.015</td><td>0.037</td><td>0.031</td><td>0.034</td><td>0.056</td></tr><tr><td></td><td>Pow3R-SLAM</td><td>0.009</td><td>0.022</td><td>0.018</td><td>0.020</td><td>0.027</td></tr><tr><td></td><td>Pow3R-SLAM hyb.</td><td>0.010</td><td>0.023</td><td>0.019</td><td>0.021</td><td>0.028</td></tr></table>

## D. Ablations

To study the effects of each of Pow3R-SLAM’s components, Table IV switches them on one at a time. The single most impactful change is the introduction of priors to Pow3R, which takes the mean ATE from 0.083 m (against MASt3R-SLAM’s 0.030 m) to 0.032 m (L1 to L3). It is the depth conditioning (not intrinsics) that plays the larger role (L2 vs. L3), and removing it (L6) regresses 20/23 scenes it still completes, while fr1/rpy diverges. Scale lowers the Sim(3) ATE further (0.032 to 0.026 m) and cuts the unscaled SE(3) error from 0.364 to 0.034 m and the map Chamfer distance from 6.6 to 3.1 cm. The anchor is neutral on this panel (at most 4.9 mm of Sim(3) ATE on any scene). Map-only keyframes keep trajectories identical while improving the Chamfer distance (3.27 to 3.05 cm with 4.4× the points). Hybrid mode gives comparable performance to the default mode, with a better median ATE, though scored per frame it is slightly worse (Sec. IV-A).

We also study the sensitivity of our methods to depth quality. Scaling each valid depth pixel by max(1+σε,0.05), $\varepsilon \sim \mathcal { N } ( 0 , 1 )$ , at all three sites raises the 24-scene mean ATE (0.026 m unperturbed) monotonically, to 0.034, 0.056, 0.197 (one sequence diverging), and 0.609 m for $\sigma = 0 . 0 5 , 0 . 1 , 0 . 2 .$ and 0.4. Noisy depth beats no depth (L2) at $\sigma = 0 . 1$ , but not at 0.2. Keeping a random 50, 10, or 3% of the depth pixels at the conditioning site instead gives 0.026, 0.028, and 0.026 m, which are within 2.2 mm of the unperturbed value. Thus, we show that the prior is robust to sparse depth, but not to dense, wrong depth. To test how depth image holes were impacting Pow3R-SLAM, we preprocessed the datasets with a depth completion network (Any2Full [27]). It regressed Replica (+9.6% mean ATE), was within noise on TUM (−1.8%), and helped 7-Scenes (−13.8%), but as it added compute cost it was not adopted. Completed pixels were less view-consistent than measured ones, and Pow3R already fills holes (Fig. 1).

## TABLE IV

ABLATIONS. MEANS AND MEDIAN ATE OVER THE 24 SCENES, CHAMFER (CM) OVER THE 15 MAP SCENES, WITH stairs SEPARATE. ROWS L1–L5 ADD ONE COMPONENT AT A TIME UP TO THE SHIPPED   
DEFAULT (DEF.); L6 AND L7 LEAVE OUT DEPTH SITES (23 SCENES, AS   
FR1/RPY DIVERGES WITHOUT COND). REF.: REFERENCE SYSTEMS, NOT   
RANKED (ORB-SLAM3 OVER ITS 21 COMPLETIONS). KF: KEYFRAMES.

<table><tr><td>configuration</td><td>Sim3 mean (24)↓</td><td>Sim3 median (24)↓</td><td>SE3 mean (24)↓</td><td>Chamfer (15)↓</td><td>stairs ↓</td></tr><tr><td>ORB-SLAM3 (ref., 21)</td><td>0.0342</td><td>0.0178</td><td>0.0411</td><td>一</td><td>0.045</td></tr><tr><td>MASt3R fresh (ref.)</td><td>0.0304</td><td>0.0228</td><td>0.1104</td><td>4.35</td><td>0.016</td></tr><tr><td>L1 Pow3R, no priors</td><td>0.0834</td><td>0.0449</td><td>0.4094</td><td>9.55</td><td>0.063</td></tr><tr><td>L2 + intrinsics</td><td>0.0711</td><td>0.0431</td><td>0.4058</td><td>8.70</td><td>0.052</td></tr><tr><td>L3 + cond</td><td>0.0319</td><td>0.0189</td><td>0.3640</td><td>6.58</td><td>0.115</td></tr><tr><td>L4 + scale</td><td>0.0260</td><td>0.0191</td><td>0.0337</td><td>3.07</td><td>0.044</td></tr><tr><td>L5 + anchor (def.)</td><td>0.0257</td><td>0.0191</td><td>0.0329</td><td>3.05</td><td>0.044</td></tr><tr><td>L6 def. − cond (23)</td><td>0.0577</td><td>0.0292</td><td>0.0658</td><td>5.56</td><td>0.066</td></tr><tr><td>L7 scale only (23)</td><td>0.0547</td><td>0.0370</td><td>0.0705</td><td>5.40</td><td>0.033</td></tr><tr><td>map-only KF off</td><td>0.0257</td><td>0.0191</td><td>0.0329</td><td>3.27</td><td>0.044</td></tr><tr><td>hybrid</td><td>0.0264</td><td>0.0173</td><td>0.0314</td><td>3.15</td><td>0.062</td></tr></table>

## V. LIMITATIONS

While the depth prior improves the prediction where sensor depth has holes, makes it metric, and improves aggregate results, it does not make the estimators more robust. Scenes for which Pow3R-SLAM is worse than MASt3R-SLAM include stairs in 7-Scenes (Fig. 4), and planar 3, sofa 2, and repetitive in ETH3D-SLAM. We hypothesize that repeating, self-similar structures cause these failures, as when Pow3R predicts a complete and confident geometry, the matching can find incorrect but mutually consistent correspondences that pull the backend to a wrong solution.

## VI. CONCLUSIONS

We presented Pow3R-SLAM, an RGB-D SLAM system that combines measured depth with learned two-view reconstruction to produce metric trajectories and dense maps. Depth conditions network predictions, sets pointmap scale, and anchors keyframe geometry during global optimization. Across 24 sequences from TUM, 7-Scenes, and Replica, Pow3R-SLAM runs 1.6× faster than MASt3R-SLAM, with 15% lower mean keyframe trajectory error and 3.1× lower per-frame trajectory error without scale alignment. Its maps are denser, and achieve 30% lower Chamfer distance on 7-Scenes and Replica. The hybrid ICP variant further improves runtime, reaching 25.3 FPS and a 2.1× speedup over MASt3R-SLAM.

Ablations identify depth conditioning as the main source of accuracy gains and scaling as essential for metric output. Conditioning remains effective with only 3% of depth pixels supplied to the network, showing that sparse measurements can still guide dense reconstruction. Depth noise and selfsimilar scenes remain challenging. Future work will investigate consistency-aware confidence gating, training with realistic sensor noise, and multi-view reconstruction priors. These results establish depth-conditioned pointmaps as a practical basis for real-time SLAM, combining improved trajectory accuracy and map completeness with lower runtime.

## ACKNOWLEDGMENT

We would like to thank the authors of MASt3R-SLAM for providing a strong and extensible baseline SLAM system. We would also like to thank the authors of Pow3R, for their strong work on integrating depth.

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC).

AI Disclosure: Claude (Fable 5.1, Opus 5) was used to assist in development and debugging of the code for this work, including organizing the LaTeX for all Tables and Fig. 2, and the Python code to stitch the images for Figs. 3 and 4. It was also used to help create the website and supplementary video. All output has been manually validated by the authors.

## REFERENCES

[1] R. A. Newcombe, A. Fitzgibbon, S. Izadi, O. Hilliges, D. Molyneaux, D. Kim, A. J. Davison, P. Kohli, J. Shotton, and S. Hodges, “Kinect-Fusion: Real-time dense surface mapping and tracking,” 2011 10th IEEE International Symposium on Mixed and Augmented Reality, pp. 127–136, 2011.

[2] T. Whelan, S. Leutenegger, R. F. Salas-Moreno, B. Glocker, and A. J. Davison, “ElasticFusion: Dense SLAM without a pose graph,” Robotics: Science and Systems, 2015.

[3] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, vol. 1, pp. 573–580, 2012.

[4] S. Wang, V. Leroy, Y. Cabon, B. Chidlovskii, and J. Revaud, “DUSt3R: Geometric 3D vision made easy,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20 697–20 709, 2024.

[5] V. Leroy, Y. Cabon, and J. Revaud, “Grounding image matching in 3D with MASt3R,” European Conference on Computer Vision, pp. 71–91, 2024.

[6] W. Jang, P. Weinzaepfel, V. Leroy, L. Agapito, and J. Revaud, “Pow3R: Empowering unconstrained 3D reconstruction with camera and scene priors,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1071–1081, 2025.

[7] R. Murai, E. Dexheimer, and A. J. Davison, “MASt3R-SLAM: Realtime dense SLAM with 3D reconstruction priors,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16 695–16 705, 2025.

[8] R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM: A´ versatile and accurate monocular SLAM system,” IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147–1163, 2015.

[9] R. A. Newcombe, S. J. Lovegrove, and A. J. Davison, “DTAM: Dense tracking and mapping in real-time,” 2011 International Conference on Computer Vision, vol. 1, pp. 2320–2327, 2011.

[10] R. Mur-Artal and J. D. Tardos, “ORB-SLAM2: An open-source´ SLAM system for monocular, stereo, and RGB-D cameras,” IEEE Transactions on Robotics, vol. 33, no. 5, pp. 1255–1262, 2016.

[11] C. Campos, R. Elvira, J. J. G. Rodr´ıguez, J. M. M. Montiel, and J. D. Tardos, “ORB-SLAM3: An accurate open-source library for visual,´ visual-inertial, and multimap SLAM,” IEEE Transactions on Robotics, vol. 37, no. 6, pp. 1874–1890, 2021.

[12] Z. Teed and J. Deng, “DROID-SLAM: deep visual SLAM for monocular, stereo, and RGB-D cameras,” in Advances in Neural Information Processing Systems (NeurIPS), 2021.

[13] E. Sucar, S. Liu, J. Ortiz, and A. J. Davison, “iMAP: Implicit mapping and positioning in real-time,” 2021 IEEE/CVF International Conference on Computer Vision (ICCV), pp. 6209–6218, 2021.

[14] Z. Zhu, S. Peng, V. Larsson, W. Xu, H. Bao, Z. Cui, M. R. Oswald, and M. Pollefeys, “NICE-SLAM: Neural implicit scalable encoding for SLAM,” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12 776–12 786, 2022.

[15] C. Yan, D. Qu, D. Xu, B. Zhao, Z. Wang, D. Wang, and X. Li, “GS-SLAM: Dense visual SLAM with 3D Gaussian splatting,” 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 19 595–19 604, 2024.

[16] J. Wang, M. Chen, N. Karaev, A. Vedaldi, C. Rupprecht, and D. Novotny, “VGGT: Visual geometry grounded transformer,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 5294–5306, 2025.

[17] G. Tolias, Y. Avrithis, and H. Jegou, “To aggregate or not to aggregate:´ Selective match kernels for image search,” 2013 IEEE International Conference on Computer Vision, pp. 1401–1408, 2013.

[18] D. Maggio, H. Lim, and L. Carlone, “VGGT-SLAM: Dense RGB SLAM optimized on the SL(4) manifold,” Advances in Neural Information Processing Systems, vol. 39, 2025.

[19] Y. Liu, S. Dong, S. Wang, Y. Yin, Y. Yang, Q. Fan, and B. Chen, “SLAM3R: Real-time dense scene reconstruction from monocular RGB videos,” 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 16 651–16 662, 2025.

[20] G. Tolias, T. Jenicek, and O. Chum, “Learning and aggregating deep local descriptors for instance-level recognition,” European Conference on Computer Vision, pp. 460–477, 2020.

[21] Y. Chen and G. Medioni, “Object modelling by registration of multiple range images,” Image and Vision Computing, vol. 10, no. 3, pp. 145– 155, 1992.

[22] S. Rusinkiewicz and M. Levoy, “Efficient variants of the ICP algorithm,” in Third International Conference on 3D Digital Imaging and Modeling, 2001.

[23] B. Glocker, S. Izadi, J. Shotton, and A. Criminisi, “Real-time RGB-D camera relocalization,” 2013 IEEE International Symposium on Mixed and Augmented Reality (ISMAR), pp. 173–179, 2013.

[24] J. Straub, T. Whelan, L. Ma, Y. Chen, E. Wijmans, S. Green, J. J. Engel, R. Mur-Artal, C. Ren, S. Verma, A. Clarkson, M. Yan, B. Budge, Y. Yan, X. Pan, J. Yon, Y. Zou, K. Leon, N. Carter, J. Briales, T. Gillingham, E. Mueggler, L. Pesqueira, M. Savva, D. Batra, H. M. Strasdat, R. D. Nardi, M. Goesele, S. Lovegrove, and R. Newcombe, “The Replica dataset: A digital replica of indoor spaces,” arXiv, 2019.

[25] T. Schops, T. Sattler, and M. Pollefeys, “BAD SLAM: Bundle adjusted¨ direct RGB-D SLAM,” 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 134–144, 2019.

[26] J. L. Schonberger and J.-M. Frahm, “Structure-from-Motion revisited,”¨ 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4104–4113, 2016.

[27] Z. Zhou, R. Liu, T. Liu, W. Zuo, S. Wang, Z. Hong, and D. Zhang, “Any to full: Prompting Depth Anything for depth completion in one stage,” arXiv, 2026.