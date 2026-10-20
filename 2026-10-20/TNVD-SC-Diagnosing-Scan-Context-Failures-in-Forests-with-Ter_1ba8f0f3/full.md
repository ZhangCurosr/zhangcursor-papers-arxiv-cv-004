# TNVD-SC: Diagnosing Scan Context Failures in Forests with Terrain-Normalized Vertical Distributions

Jineon Kim<sup>1</sup> and Jhonghyun An<sup>1,∗</sup>

Abstract— Long-term LiDAR place recognition in forests is difficult because repetitive vegetation and uneven terrain create severe structural aliasing. We diagnose Scan Context (SC) by separating candidate retrieval from fine matching: on the standardized TreeLoc++ Wild-Places protocol, a TNVD support key places a valid location in the top-25 set for approximately 99% of queries, while conventional SC matching on those candidates reaches only 61.84% mean R@1. With an identical raw-SC candidate set, terrain normalization improves max-height cells by only 1.21 points but vertical-distribution cells by 19.72 points. These observations motivate Terrain-Normalized Vertical Distribution Scan Context (TNVD-SC), which preserves the SC ring–sector topology while encoding terrain-relative vertical probability distributions. Under the standardized TreeLoc++ protocol, TNVD-SC reaches 91.83% mean R@1 and attains the highest R@1 mean and R@1 stability among the compared methods, while TreeLoc++ remains stronger in mean MR, MF1, and AUC. The unchanged method reaches 85.14% mean R@1 on the four-session CVTNet-style NCLT evaluation. A lossless count-coded implementation with exact hierarchical Jensen–Shannon search reduces descriptor payload by 3.86× and compute latency by 23.8% without changing the audited retrieval outputs.

## I. INTRODUCTION

Reliable place recognition is a core component of longterm robotic autonomy, supporting loop closure, global relocalization, and recovery after localization failure. Forests are particularly difficult: tree stems repeat at similar spacings, foliage and understory vary over time, visibility changes with viewpoint and season, and terrain relief changes the apparent vertical arrangement of otherwise similar places. Wild-Places was introduced specifically to expose these challenges in unstructured natural environments [1].

Scan Context (SC) [2], [3] remains attractive for field robots because it is training-free, requires no site-specific retraining, and preserves a simple coarse-to-fine retrieval structure suitable for CPU-class deployment. In our standardized comparison (Table III), every learned baseline has an R@1 stability ratio below 1.85, whereas TNVD-SC exceeds 2.8 on both sites, making pair-to-pair consistency an additional practical consideration. A LiDAR scan is divided into polar ring–sector cells, each cell is compressed to a maximumheight scalar, a yaw-invariant key retrieves candidates, and circular sector shifts resolve yaw. The compression is severe: all vertical structure inside a cell is reduced to one extreme value. In forests, different places can therefore produce similar SC patterns whenever their upper envelopes align even though their internal vegetation layers differ.

This ambiguity is not removed simply by widening retrieval. Figure 2 shows that, even at K = 250, SC candidate recall reaches 97.45% on Venman and 98.47% on Karawatha while final R@1 remains near 40–50%. We therefore treat retrieval and fine matching as separate failure modes rather than assuming that a stronger candidate search alone will resolve structural aliasing.

Recent forest place-recognition methods use learned BEV density representations [4], hierarchical transformers [5], [6], explicit tree geometry [7], and training-free signalprocessing approaches [8]. Handcrafted descriptors have also explored richer height or probabilistic structure [9]–[11]. Thus, vertical distributions, terrain normalization, and probability divergences are not claimed as individually new. Our question is diagnostic: at which stage does SC fail in long-term forests, and what representation change is justified by that failure?

We use TNVD to denote the terrain-normalized verticaldistribution cell representation and TNVD-SC to denote the complete place-recognition system consisting of TNVD construction, support-key retrieval, and circular distribution matching. Two controlled observations motivate the system. First, the TNVD support key places a valid location in the top-25 set for 98.89% of Venman and 98.76% of Karawatha queries, yet conventional SC matching on those candidates reaches only 57.20% and 66.47% R@1. Second, when the candidate set is fixed to the same raw-SC ringkey Top-25 set, terrain normalization changes max-height mean R@1 by only +1.21 points but changes the verticaldistribution representation by +19.72 points. The benefit of terrain normalization is therefore strongly conditioned on what information the cell representation preserves.

These findings motivate TNVD-SC. The method keeps the SC range–azimuth topology and exhaustive circular yaw search, but replaces the per-cell maximum with a terrainrelative vertical probability distribution. A 480-D supportonly key is used only for coarse retrieval; the full ordered tensor is retained for fine re-ranking. The frozen method is implemented efficiently through lossless integer count coding and exact hierarchical Jensen–Shannon pruning, preserving all audited retrieval outputs while reducing memory and latency.

Our contributions are:

• Stage-wise failure diagnosis. We separate SC candidate retrieval from fine discrimination and show that retrieving a valid candidate does not by itself resolve forest aliasing.

![](images/28c2d1cb14f1272eae7a21905b7ba087c05412368a40a660b07cae207b379282.jpg)

![](images/760ec15472ab8927d520f3c6403d0a8ea072d38facc803b4bb53f1e7d68c0976.jpg)  
Fig. 1. Overview of TNVD-SC. Query and database scans share the TNVD backbone. A 480-D support-only key retrieves top-K candidates, while the ordered full TNVD tensor performs circular fine matching. Bottom: conventional SC stores one maximum height per cell, whereas TNVD retains its terrain-relative vertical distribution.

• Controlled representation–terrain interaction. With identical raw-SC candidates across all conditions, terrain normalization provides a marginal gain for max-height cells but a much larger gain for vertical-distribution cells, revealing an 18.51-point interaction in mean R@1.

• Diagnosis-driven TNVD-SC. We derive a training-free SC-compatible system with a 480-D support retrieval key and circular distribution matching, reaching 91.83% mean R@1 on standardized Wild-Places and 85.14% on the four-session CVTNet-style NCLT evaluation.

• Exact efficient implementation. Lossless count coding and exact hierarchical Jensen–Shannon search reduce persistent descriptor payload by 3.86× and compute latency by 23.8% without changing the audited retrieval decisions.

## II. RELATED WORK

## A. Handcrafted and Probabilistic LiDAR Descriptors

SC [2] introduced polar max-height cells with yaw-invariant retrieval and circular matching, while SC++ [3], RING++ [12], and SOLiD [13] pursue complementary invariances or compact encodings. Matched Filtering [8] uses a training-free BEV representation with two-stage correlation for place recognition and relative pose; its Wild-Places numbers use the original 3 m protocol and are therefore not mixed with our

5 m TreeLoc++ comparison. Richer vertical/probabilistic representations include PGHCI [10], CSC [9], and PROBE [11]. These works motivate richer structure, but our focus is the controlled diagnosis of which SC stage fails and which SCcompatible representation addresses that failure.

## B. Learned and Forest-Specific Recognition

Learned LiDAR retrieval spans PointNetVLAD, MinkLoc3D/v2, TransLoc3D, LoGG3D-Net, OverlapTransformer, SeqOT/CVTNet, BEVPlace++, and OverlapMamba [14]–[23]. Forest-specific methods include ForestLPR, HOTFormer-Loc/HOTFLoc++, and TreeLoc++ [4]–[7]. Robust terrain estimators such as Patchwork/Patchwork++ [24], [25] further show that local ground modeling is well studied. We therefore do not claim terrain estimation itself as novel; TNVD-SC uses one frozen lightweight reference to isolate its interaction with the downstream cell representation.

## III. DIAGNOSING SCAN CONTEXT IN FORESTS

## A. Retrieval and Fine Matching Are Distinct Bottlenecks

We decompose an SC-style pipeline into (i) descriptor construction, (ii) yaw-invariant candidate retrieval, and (iii) circular fine matching. A positive that never enters the candidate list and a positive that is retrieved but then loses to an aliased forest location are different failure modes.

![](images/6102672e392fdbe2b72d6e09071a7d7d76deccfd0c9f32b504be108f1a10ea8e.jpg)

![](images/e6f06aa4057e20a0ebf18f3392b0617dbcc05e89ae5e80f0d8c285edbb70bb0c.jpg)  
Fig. 2. Candidate-budget diagnosis for conventional SC. Candidate recall continues to rise with K, whereas final SC R@1 saturates. The single dashed line marks the $K = 2 5$ operating point used by TNVD-SC.

TABLE I  
COMPONENT DECOMPOSITION ON STANDARDIZED WILD-PLACES. R@25 IS CANDIDATE RECALL BEFORE FINE RERANKING.
<table><tr><td>Retrieval</td><td>Matcher</td><td colspan="2">Venman</td><td colspan="2">Karawatha</td><td>Mean</td></tr><tr><td></td><td></td><td>R@1</td><td>R@25</td><td>R@1</td><td>R@25</td><td>R@1</td></tr><tr><td>SC ring key</td><td>SC</td><td>40.53</td><td>76.32</td><td>53.37</td><td>86.82</td><td>46.95</td></tr><tr><td>SC ring key</td><td>TNVD</td><td>68.72</td><td>76.32</td><td>79.15</td><td>86.82</td><td>73.94</td></tr><tr><td>TNVD support</td><td>SC</td><td>57.20</td><td>98.89</td><td>66.47</td><td>98.76</td><td>61.84</td></tr><tr><td>TNVD support</td><td>TNVD</td><td>92.24</td><td>98.89</td><td>91.41</td><td>98.76</td><td>91.83</td></tr></table>

Figure 2 first varies the SC candidate budget. Candidate recall continues to rise as K increases, reaching 97.45% on Venman and 98.47% on Karawatha at $K = 2 5 0 ,$ , while final SC R@1 remains near 40–50%. Increasing retrieval breadth therefore does not remove the fine-matching failure. Table I then controls retrieval and matching directly. The TNVD support key places a valid location in the top-25 set for 98.89% and 98.76% of queries, yet the SC matcher on those candidates reaches only 57.20% and 66.47% R@1. Conversely, the TNVD matcher improves the fixed SC candidate set to 68.72% and 79.15% but remains bounded by its lower candidate recall. Only the full TNVD-SC combination reaches 92.24% and 91.41%. Candidate recovery and fine discrimination are therefore distinct, non-substitutable bottlenecks.

## B. Terrain Normalization Depends on the Cell Representation

To isolate terrain normalization from candidate retrieval, we generate the raw-SC ring-key Top-25 candidates once and keep that candidate set identical in all four conditions of Table II. Only the cell representation and whether the same local terrain reference is applied are changed. Terrain normalization raises mean R@1 for max-height cells only from 46.95% to 48.16% (+1.21 points), whereas the vertical-distribution representation rises from 54.22% to 73.94% (+19.72 points). The resulting representation-dependent interaction is +18.51 points.

The difference is consistent with the information retained by each representation. A max-height cell keeps only one upperenvelope scalar, so removing low-frequency terrain relief changes that already compressed cue only modestly. A vertical distribution instead preserves multiple height layers; without normalization, terrain offsets translate probability mass along the height-bin axis and can break bin correspondence across revisits. Terrain-relative coordinates align understory, stem, and canopy layers before distribution matching. Terrain normalization is therefore not an independently sufficient explanation for the gain; its value depends strongly on the representation that follows it. Because all four conditions share the raw-SC candidate set, their achievable R@1 is capped by candidate recall (76.32% on Venman and 86.82% on Karawatha). The terrain-normalized vertical-distribution matcher already reaches 90.0% and 91.2% of these ceilings, respectively, indicating that the controlled interaction measurement is itself retrieval-ceiling limited.

TABLE II  
TERRAIN NORMALIZATION UNDER AN IDENTICAL RAW-SC TOP-25 CANDIDATE SET (PAIR-MEAN R@1, %).
<table><tr><td>Cell representation</td><td>Terrain norm.</td><td>Venman</td><td>Karawatha</td><td>Mean</td></tr><tr><td>Max-height</td><td>No</td><td>40.53</td><td>53.37</td><td>46.95</td></tr><tr><td>Max-height</td><td>Yes</td><td>42.65</td><td>53.67</td><td>48.16</td></tr><tr><td>Vertical distribution</td><td>No</td><td>50.66</td><td>57.78</td><td>54.22</td></tr><tr><td>Vertical distribution</td><td>Yes</td><td>68.72</td><td>79.15</td><td>73.94</td></tr></table>

## IV. TERRAIN-NORMALIZED VERTICAL DISTRIBUTION SCAN CONTEXT

Figure 1 summarizes the diagnosis-driven design. We use TNVD for the cell-level terrain-normalized verticaldistribution representation and TNVD-SC for the complete retrieval-and-matching system. TNVD-SC retains the original SC range–azimuth partition and exhaustive circular yaw search while changing the vertical coordinate, per-cell encoding, retrieval key, and fine matcher.

## A. Lightweight Terrain-Relative Height

Let a LiDAR scan be $\mathcal { P } = \{ { \bf p } _ { i } \} _ { i = { \bf \bar { \Phi } } } ^ { N }$ with $\mathbf { p } _ { i } = ( x _ { i } , y _ { i } , z _ { i } )$ We estimate a lightweight local terrain reference $g ( x , y )$ on a Cartesian grid with 2 m cells. Cells containing at least three in-range points obtain a provisional terrain estimate from the 10th percentile of their z values. Each grid cell is then assigned the median of available provisional estimates within its $3 \times 3$ neighborhood, simultaneously smoothing valid estimates and filling sparse cells. If no valid provisional estimate exists in that neighborhood, a scan-level 10thpercentile height is used as a fallback. The terrain-relative height is

$$
h _ { i } = \operatorname* { m a x } \left( 0 , \ z _ { i } - g ( x _ { i } , y _ { i } ) \right) .\tag{1}
$$

The fixed parameters are engineering choices: the 2 m grid retains local relief while limiting sparsity, the low percentile is more robust to vegetation than a single minimum, the threepoint minimum avoids extremely small local samples, and the $3 \times 3$ median/fill suppresses isolated outliers and fills local holes. More sophisticated LiDAR ground estimators exist [24], [25]; ground segmentation itself is not our contribution.

An audit confirms that the global fallback has negligible influence on the descriptor-relevant points: it is used for 0.000248% on Wild-Places, 0.000007% on Oxford Forest, and 0.060355% on NCLT. Even on NCLT, the median and 95th-percentile per-scan fallback ratios are only 0.0583% and 0.1850%, respectively. The same estimator and parameters are frozen in every terrain-normalized experiment.

Figure 3 illustrates the role of this normalization: lowfrequency terrain relief is removed from the vertical coordinate while local vegetation structure is preserved for the subsequent distribution encoding.

![](images/5b278b6987167cdf0c83e216659b38d14bce1fc47af5664e089078094153077e.jpg)

![](images/952cd06b551cf45b9ab26ea3450a98dbb1191d81238d6d65a5f07b1dbdb90f12.jpg)

![](images/eeca11ef1963d10d1d7d2f7ec66fd9d85979de4afcd021543434006ca1384cd9.jpg)  
Fig. 3. Effect of terrain normalization on the vertical representation. The same local forest corridor is shown in absolute-height and terrain-relative coordinates. Removing low-frequency terrain relief aligns the local vertical reference while preserving vegetation-layer structure, which is subsequentl encoded by TNVD.

## B. Polar Vertical Distribution

As in ${ \mathrm { S C } } ,$ the horizontal plane is divided into $N _ { R } = 2 0$ rings and $N _ { S } = 6 0$ sectors within an 80 m radius. For a point assigned to cell $( r , s )$ , terrain-relative height is quantized into $B = 2 4$ bins over $[ 0 , H ]$ , with $H = 1 8$ m and overflow accumulated in the final bin:

$$
b ( h _ { i } ) = \operatorname* { m i n } \biggl ( B - 1 , \left\lfloor \frac { h _ { i } } { H / B } \right\rfloor \biggr ) .\tag{2}
$$

Let $n _ { r , s , b }$ be the number of points in vertical bin b and $\begin{array} { r } { N _ { r , s } = \sum _ { b = 0 } ^ { B - 1 } n _ { r , s , b } } \end{array}$ . We define the observation mask $O _ { r , s } =$ $\mathbb { I } [ N _ { r , s } > 0 ]$ and

$$
D _ { r , s , b } = \left\{ \begin{array} { l l } { \frac { n _ { r , s , b } } { N _ { r , s } } , } & { O _ { r , s } = 1 , } \\ { 0 , } & { O _ { r , s } = 0 , } \end{array} \right. \ b = 0 , \ldots , B - 1 .\tag{3}
$$

Thus $D _ { r , s , }$ <sub>:</sub> is a categorical distribution only for an observed cell; an unobserved cell is represented by the zero vector and excluded from distribution-to-distribution comparison. TNVD preserves the relative probability mass across vertical layers while normalizing away the absolute return count. Section V-E verifies that the probability mass contains information beyond binary vertical occupancy.

## C. Compact Yaw-Invariant Support Key

For coarse retrieval, we summarize whether each vertical bin is present across the sectors of a ring:

$$
u _ { r , b } = \frac { 1 } { N _ { S } } \sum _ { s = 0 } ^ { N _ { S } - 1 } \mathbb { I } [ n _ { r , s , b } > 0 ] .\tag{4}
$$

The indicator is deliberately binary: a sector contributes one whenever at least one point occupies bin $b ,$ independent of the number of returns. Vectorizing $u _ { r , b }$ produces a $N _ { R } B = 4 8 0$ dimensional key. Because the sector axis is aggregated, the key is invariant to circular shifts and, more strongly, arbitrary sector permutations. We accept this stronger invariance only at coarse retrieval; the full TNVD tensor retains sector ordering for candidate disambiguation. A KD-tree with Euclidean key distance retrieves $K = 2 5$ candidates.

## D. Circular Jensen–Shannon Matching

For two observed cell distributions $P , Q \ \in \ \Delta ^ { B - 1 }$ , let $M = ( P + Q ) / 2$ . Using the natural logarithm,

$$
\mathrm { J S D } ( P , Q ) = \frac { 1 } { 2 } \mathrm { K L } ( P \| M ) + \frac { 1 } { 2 } \mathrm { K L } ( Q \| M ) ,\tag{5}
$$

where $\begin{array} { r } { \mathrm { K L } ( P \| M ) = \sum _ { b : P _ { b } > 0 } P _ { b } \ln ( P _ { b } / M _ { b } ) } \end{array}$ . Hence $0 ~ \leq$ $\mathrm { J S D } ( P , Q ) \leq \ln 2 .$ , and the normalized cell similarity is

$$
c ( P , Q ) = 1 - \frac { \mathrm { J S D } ( P , Q ) } { \ln 2 } \in [ 0 , 1 ] .\tag{6}
$$

For sector shift $\tau \in \{ 0 , \ldots , N _ { S } - 1 \}$ , the database tensor is circularly shifted to the right,

$$
\begin{array} { r l } & { D _ { r , s , b } ^ { d , ( \tau ) } = D _ { r , ( s - \tau ) \mathrm { ~ m o d ~ } N _ { S } , b } ^ { d } , } \\ & { O _ { r , s } ^ { d , ( \tau ) } = O _ { r , ( s - \tau ) \mathrm { ~ m o d ~ } N _ { S } } ^ { d } . } \end{array}\tag{7}
$$

Define the active union

$$
\mathcal { A } _ { \tau } = \{ ( r , s ) : O _ { r , s } ^ { q } = 1 \ \vee \ O _ { r , s } ^ { d , ( \tau ) } = 1 \} .\tag{8}
$$

The active-cell contribution is

$$
\gamma _ { r , s } ^ { ( \tau ) } = \left\{ \begin{array} { l l } { c \Big ( D _ { r , s , : } ^ { q } , D _ { r , s , : } ^ { d , ( \tau ) } \Big ) , } & { O _ { r , s } ^ { q } = O _ { r , s } ^ { d , ( \tau ) } = 1 , } \\ { 0 , } & { O _ { r , s } ^ { q } \neq O _ { r , s } ^ { d , ( \tau ) } . } \end{array} \right.\tag{9}
$$

Both-empty cells are absent from $\boldsymbol { A } _ { \tau } \dot { \boldsymbol { : } }$ ; one-sided cells enlarge the union but contribute zero. The shift score is

$$
S ( q , d ; \tau ) = \frac { 1 } { | \mathcal { A } _ { \tau } | } \sum _ { ( r , s ) \in \mathcal { A } _ { \tau } } \gamma _ { r , s } ^ { ( \tau ) } .\tag{10}
$$

Finally,

$$
\hat { \tau } = \underset { 0 \leq \tau < N _ { S } } { \arg \operatorname* { m a x } } S ( q , d ; \tau ) , \qquad S ^ { * } ( q , d ) = S ( q , d ; \hat { \tau } ) .\tag{11a}
$$

$$
\hat { \psi } _ { q  d } = - \hat { \tau } \frac { 2 \pi } { N _ { S } } \quad ( \mathrm { m o d } 2 \pi ) .\tag{11b}
$$

Here $\hat { \psi } _ { q  d }$ denotes the discrete query-to-database circular alignment associated with the best descriptor match under the stated sector-index convention. With $N _ { S } = 6 0 $ , one sector corresponds to $6 ^ { \circ }$ . The candidate with the largest $S ^ { * }$ is returned as the recognized place.

TABLE III  
INTER-SESSION PLACE RECOGNITION PERFORMANCE ON WILD-PLACES UNDER THE STANDARDIZED TREELOC++ TABLE-5 PROTOCOL. EACH SITECONTAINS 12 DIRECTED QUERY–DATABASE PAIRS. FOR EACH METRIC, WE REPORT THE PAIR MEAN AND STABILITY RATIO S FROM EQ. (12).PUBLISHED BASELINES ARE FROM TREELOC++ [7]. BEST VALUES ARE BOLD.
<table><tr><td rowspan="3">Category</td><td rowspan="3">Method</td><td colspan="7">Venman01–04 (12 query-database pairs)</td><td colspan="8">Karawatha01–04 (12 query-database pairs)</td></tr><tr><td colspan="2">R@1</td><td colspan="2">MR</td><td colspan="2">MF1</td><td colspan="2">AUC</td><td colspan="2">R@1</td><td colspan="2">MR</td><td colspan="2">MF1</td><td colspan="2">AUC</td></tr><tr><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td><td>Mean</td><td>S</td></tr><tr><td rowspan="6">Learning</td><td>TransLoc3D</td><td>.553</td><td>.464</td><td>.008</td><td>-.048 .707</td><td>.678</td><td></td><td>.604 .511</td><td></td><td>.711</td><td>1.316</td><td>.013</td><td>-.103</td><td>.831</td><td>1.561</td><td>.756 .921</td><td>1.268</td></tr><tr><td>LoGG3D-Net</td><td>.660</td><td>1.242</td><td>.003</td><td>-.282</td><td>.813</td><td>1.382</td><td>.732</td><td>.872</td><td>.897</td><td>1.848</td><td>.002</td><td>-.278</td><td>.948</td><td>2.299</td><td></td><td>1.638</td></tr><tr><td>MinkLoc3Dv2</td><td>.755</td><td>.782</td><td>.033</td><td>.161</td><td>.857</td><td>1.030</td><td>.818</td><td>.881</td><td>.849</td><td>1.607</td><td>.062</td><td>.150</td><td>.919</td><td>1.889</td><td>.911</td><td>1.592</td></tr><tr><td>BEVPlace++</td><td>.735</td><td>.701</td><td>.091</td><td>-.025</td><td>.840</td><td>.920</td><td>.841</td><td>.842</td><td>.767</td><td>.934</td><td>.016</td><td>-.193</td><td>.868</td><td>1.179</td><td>.767</td><td>.822</td></tr><tr><td>ForestLPR</td><td>.718</td><td>.695</td><td>.023</td><td>.006</td><td>.842</td><td>1.006</td><td>.839</td><td>1.010</td><td>.790</td><td>1.136</td><td>.072</td><td>.024</td><td>.885</td><td>1.366</td><td>.885</td><td>1.205</td></tr><tr><td>HOTFormerLoc</td><td>.899</td><td>1.173</td><td>.155</td><td>-.057</td><td>.946</td><td>1.448</td><td>.938</td><td>1.350</td><td>.885</td><td>1.341</td><td>.088</td><td>.026</td><td>.941</td><td>1.628</td><td>.937</td><td>1.471</td></tr><tr><td rowspan="7">Handcrafted</td><td>SC++</td><td>.245</td><td>.429</td><td>.056</td><td>.129</td><td>.425</td><td>.631</td><td>.396</td><td>.430</td><td>.370</td><td>.632</td><td>.117</td><td>.146</td><td>.560</td><td>.817</td><td>.564</td><td>.704</td></tr><tr><td>SOLiD</td><td>.150</td><td>.199</td><td>.014</td><td>-.263</td><td>.288</td><td>.321</td><td>.241</td><td>.093</td><td>.158</td><td>.357</td><td>.000</td><td>.000</td><td>.300</td><td>.501</td><td>.211</td><td>.318</td></tr><tr><td>RING++</td><td>.322</td><td>.448</td><td>.057</td><td>-.004</td><td>.503</td><td>.624</td><td>.500</td><td>.458</td><td>.341</td><td>.713</td><td>.030</td><td>-.143</td><td>.509</td><td>.848</td><td>.402</td><td>.439</td></tr><tr><td>BTC</td><td>.185</td><td>.543</td><td>.006</td><td>-.337</td><td>.331</td><td>.594</td><td>.216</td><td>.338</td><td>.068</td><td>.169</td><td>.005</td><td>-.193</td><td>.221</td><td>.150</td><td>.128</td><td>.024</td></tr><tr><td>MapClosure</td><td>.478</td><td>.338</td><td>.054</td><td>-.008</td><td>.618</td><td>.459</td><td>.465</td><td>.279</td><td>.484</td><td>.420</td><td>.036</td><td>-.121</td><td>.641</td><td>.571</td><td>.479</td><td>.347</td></tr><tr><td>TreeLoc</td><td>.766</td><td>.940</td><td>.046</td><td>-.019</td><td>.890</td><td>1.202</td><td>.853</td><td>.954</td><td>.486</td><td>.938</td><td>.003</td><td>.785</td><td>.858</td><td>1.747</td><td>.838</td><td>1.577</td></tr><tr><td>TreeLoc++</td><td>.896</td><td>1.207</td><td>.970</td><td>1.469</td><td>.993</td><td>2.303</td><td>.998</td><td>3.281</td><td>.728</td><td>1.266</td><td>.849</td><td>1.008</td><td>.974</td><td>2.149</td><td>.995</td><td>2.801</td></tr><tr><td colspan="2">TNVD-SC (ours)</td><td>.922</td><td>2.822</td><td>.364</td><td>.329</td><td>.960</td><td>3.473</td><td>.973</td><td>3.999</td><td>.914</td><td>3.329</td><td>.236</td><td>.395</td><td>.956</td><td>4.012</td><td>.970</td><td>4.094</td></tr></table>

R@1: Recall@1; MR: maximum recall at 100% precision; MF1: maximum F1 score; AUC: area under the precision–recall curve; S: stability ratio. TABLE IV 1 1

## V. EXPERIMENTAL EVALUATION

## A. Datasets and Protocols

One place is one input scan and one global descriptor for our SC and TNVD-SC rows; neither uses temporal accumulation. Wild-Places: we follow the TreeLoc++ intersession protocol [7] on Venman01–04 and Karawatha01–04 (12 directed pairs/site), with the predefined test region, the full paired session as database, a 5 m horizontal positive radius, and pair-wise means. NCLT [26]: we reproduce the released SeqOT/CVTNet full-database evaluator [20], [21]; 2012-01-08 (28,127 scans) is the database, four long-term sessions are queried every five scans, and positives use a 15 m pose threshold. Oxford Forest: Evo, Forest of Dean, Stein am Rhein, and Wytham use the released TreeLoc++ Oxford data with a 5 m positive radius and 50-frame exclusion. TreeLoc++ is re-run under the same evaluation rules and evaluated-query cohorts (1361/451/491/98). The frozen Wild-Places configuration B = 24, H = 18 m, $N _ { R } = 2 0$ $N _ { S } =$ 60, K = 25 is reused unchanged on NCLT and Oxford.

## B. Wild-Places: TreeLoc++ Standardized Comparison

We follow the exact TreeLoc++ Table-5 reporting protocol [7]. For each site, the four sessions form 12 directed inter-session query–database pairs. Each metric is reported by its pair mean $\mu$ and the TreeLoc++ stability ratio

$$
S = \log \left( \frac { \mu } { \sigma + \epsilon } \right) ,\tag{12}
$$

where σ is the population standard deviation over the 12 directed pairs and $\epsilon = 1 0 ^ { - 1 2 }$ is the numerical stabilizer used by the Table-5 metric computation. For TNVD-SC R@1, the corresponding pair-wise standard deviations are approximately 0.0548 and 0.0327 on Venman and Karawatha, respectively, so ϵ is negligible relative to the measured variation. Table III shows that TNVD-SC has the highest R@1 mean and stability on both sites, while TreeLoc++ remains stronger in mean MR, MF1, and AUC; we therefore claim top-1 accuracy/consistency rather than all-metric dominance.

NCLT FOUR-SESSION CVTNET EVALUATION (R@1, %). PUBLISHED BASELINES ARE FROM CVTNET [21]; OUR ROWS USE THE RELEASED FULL-DATABASE EVALUATOR.
<table><tr><td>Method</td><td>02-05 06-1502-23</td><td></td><td>04-05 Mean</td></tr><tr><td>PointNetVLAD</td><td>74.6 61.2</td><td>46.9</td><td>44.9 56.90</td></tr><tr><td>MinkLoc3D</td><td>80.2 63.0</td><td>50.7 48.2</td><td>60.53</td></tr><tr><td>OverlapTransformer</td><td>86.1 63.9</td><td>53.6</td><td>49.6 63.30</td></tr><tr><td>SeqOT</td><td>91.7 76.2</td><td>69.1</td><td>63.9 75.23</td></tr><tr><td>CVTNet</td><td>93.2 79.3</td><td>81.7</td><td>78.0 83.05</td></tr><tr><td>SC (ours)</td><td>91.89 69.09</td><td>70.15</td><td>63.77 73.73</td></tr><tr><td>TNVD-SC</td><td>93.38 81.61</td><td>84.59</td><td>80.97 85.14</td></tr></table>

Among 12,350 top-1 true positives with query and database pose orientations available, the discrete circular alignment has a median absolute yaw error of 2.33<sup>◦</sup> and a 90th percentile of 8.29<sup>◦</sup> relative to the ground-truth query-todatabase heading, supporting the geometric interpretation of the maximizing shift. The MR gap is mainly confidence separability: TreeLoc++ geometry-overlap confidence yields correctness ROC-AUC 0.998/0.990 (Venman/Karawatha), versus 0.790/0.768 for TNVD-SC similarity, motivating selective structural verification. Matched Filtering [8] and HOTFLoc++ [6] use different 3 m Wild-Places protocols and are not mixed into Table III.

## C. NCLT Long-Term Generalization

Table IV evaluates the unchanged Wild-Places configuration with the released SeqOT/CVTNet full-database protocol. TNVD-SC improves our full-yaw SC baseline on every date, raising mean R@1 from 73.73% to 85.14% (+11.41 pp) and exceeding CVTNet’s reported four-session mean by 2.09 points. This claim is restricted to the reproduced CVTNet/SeqOT setup because other NCLT studies vary sessions, context, modality, or thresholds.

## D. Oxford Forest Generalization

Table V adds a TreeLoc++ re-run to the frozen Oxford control under the same 5 m / 50-frame evaluated-query cohorts. TNVD-SC improves SC on all four sequences (24.12% to

TABLE V  
OXFORD FOREST INTRA-SESSION R@1 (%) UNDER THE COMMON 5 M / 50-FRAME PROTOCOL. TREELOC++ IS RE-RUN ON THE SAME RELEASED SEQUENCES AND EVALUATED-QUERY COHORTS.
<table><tr><td>Sequence</td><td>SC</td><td>TNVD-SC</td><td>TreeLoc++</td><td>∆</td></tr><tr><td>Evo</td><td>30.49</td><td>40.78</td><td>93.90</td><td>+10.29</td></tr><tr><td>Forest of Dean</td><td>22.39</td><td>73.17</td><td>76.27</td><td>+50.78</td></tr><tr><td>Stein am Rhein</td><td>35.44</td><td>45.42</td><td>93.69</td><td>+9.98</td></tr><tr><td>Wytham</td><td>8.16</td><td>68.37</td><td>39.80</td><td>+60.21</td></tr><tr><td>Mean</td><td>24.12</td><td>56.94</td><td>75.92</td><td>+32.82</td></tr></table>

TABLE VI

RETRIEVAL-KEY, SIMILARITY, AND REPRESENTATION ABLATIONS ON STANDARDIZED WILD-PLACES (R@1, %). THE FULL-DISTRIBUTION COSINE ENTRY INTENTIONALLY REPEATS THE COSINE-SIMILARITY ROW AS THE REFERENCE CONDITION FOR THE BINARY-REPRESENTATION ABLATION.
<table><tr><td>Ablation</td><td>Setting</td><td></td><td>Venman Karawatha Mean</td><td></td></tr><tr><td rowspan="5">Retrieval key</td><td>Mean, 480-D</td><td>89.34</td><td>89.59</td><td>89.47</td></tr><tr><td>Support only, 480-D</td><td>92.24</td><td>91.41</td><td>91.83</td></tr><tr><td>Mean+Std, 960-D</td><td>88.04</td><td>87.82</td><td>87.93</td></tr><tr><td>Mean+Support, 960-D</td><td>92.31</td><td>91.40</td><td>91.86</td></tr><tr><td>Mean+Std+Support, 1440-D</td><td>92.37</td><td>91.36</td><td>91.87</td></tr><tr><td rowspan="3">Similarity</td><td>Cosine</td><td>92.38</td><td>91.30</td><td>91.84</td></tr><tr><td>Bhattacharyya</td><td>90.87</td><td>90.07</td><td>90.47</td></tr><tr><td>Jensen-Shannon</td><td>92.24</td><td>91.41</td><td>91.83</td></tr><tr><td rowspan="2">Representation</td><td>Binary + cosine</td><td>82.72</td><td>86.76</td><td>84.74</td></tr><tr><td>Full dist. + cosine</td><td>92.38</td><td>91.30</td><td>91.84</td></tr></table>

TABLE VII

VERTICAL DISCRETIZATION SENSITIVITY ON STANDARDIZED WILD-PLACES (R@1, %).
<table><tr><td>Setting</td><td></td><td>Value Key dim. Venman</td><td></td><td>Karawatha / Mean</td></tr><tr><td rowspan="4">B (H = 18 m)</td><td>6</td><td>120</td><td>81.66</td><td>81.87  / 81.77</td></tr><tr><td>12</td><td>240</td><td>89.21</td><td>88.60 / 88.91</td></tr><tr><td>24</td><td>480</td><td>92.24</td><td>91.41 / 91.83</td></tr><tr><td>36</td><td>720</td><td>93.34</td><td>92.39 / 92.87</td></tr><tr><td rowspan="4">H (B = 24)</td><td>12 m</td><td>480</td><td>92.32</td><td>89.90 / 91.11</td></tr><tr><td>18 m</td><td>480</td><td>92.24</td><td>91.41 / 91.83</td></tr><tr><td>24 m</td><td>480</td><td>91.62</td><td>91.03 / 91.33</td></tr><tr><td>30 m</td><td>480</td><td>90.58</td><td>90.37  / 90.48</td></tr></table>

56.94% mean), while TreeLoc++ leads overall at 75.92%; however, TNVD-SC is close on Forest of Dean (73.17% vs. 76.27%) and higher on Wytham (68.37% vs. 39.80%), indicating complementary site-wise behavior.

## E. Ablations and Design Choices

Table VI separates the requirements of coarse retrieval from those of fine matching. The 480-D support-only key reaches 91.83% mean R@1, only 0.04 points below the 1440- D mean+std+support key, indicating that occupancy across vertical bins is already sufficient for candidate generation. In contrast, the ordered matcher benefits strongly from retaining probability mass: binarizing the full distribution lowers mean R@1 from 91.84% to 84.74%. Cosine and Jensen–Shannon matching are nearly tied in raw accuracy (91.84% versus 91.83%), so Jensen–Shannon is not selected because of an isolated accuracy advantage. We use it because the cell descriptor is a normalized distribution and because its divergence structure supports the exact hierarchical pruning

TABLE VIII  
IMPLEMENTATION EFFICIENCY. SC TIMING IS REPORTED AS A REFERENCE FROM THE SAME K = 25, 500-QUERY CLEAN BENCHMARK;  
ITS DB-RSS VALUE WAS NOT RECORDED WITH THE FINAL COMPACT-MEMORY CAMPAIGN.
<table><tr><td>Implementation</td><td>Payload [kB] DB [MiB] Query [ms] Rate [Hz]</td><td></td><td></td><td></td></tr><tr><td>SC reference</td><td>9.7</td><td></td><td>18.73</td><td>53.4</td></tr><tr><td>TNVD float exhaustive</td><td>239.0</td><td>504.9</td><td>228.55</td><td>4.38</td></tr><tr><td>TNVD count + exact hier.</td><td>61.9</td><td>188.5</td><td>174.27</td><td>5.74</td></tr></table>

described below.

Table VII further shows that most of the discretization gain is obtained before B = 24. Increasing B from 6 to 12 and then 24 raises mean R@1 from 81.77% to 88.91% and 91.83%, whereas B = 36 adds only 1.04 points while increasing retrieval-key dimension by 50%. We therefore retain $B = 2 4$ as the accuracy–compactness knee. With B fixed, performance is comparatively stable around H = 18 m and degrades as the range becomes too large, consistent with a fixed bin budget allocating less resolution to the vegetation layers that are most discriminative. These ablations freeze $B \ = \ 2 4$ $H \ = \ 1 8 \mathbf { m }$ , support-only retrieval, and full-distribution Jensen–Shannon matching for all reported generalization experiments.

## F. Implementation Efficiency

The TNVD tensor is reconstructed exactly from uint16 bin counts, reducing persistent payload from 239,040 to 61,920 bytes/place (3.86×). This representation is lossless with respect to the normalized TNVD probabilities because each probability is recovered from the stored per-bin counts and the corresponding cell total; no quantized probability approximation is introduced.

Exact hierarchical Jensen–Shannon search reduces matching cost without changing the objective. Adjacent vertical bins are first aggregated into a coarser distribution and used only to reject yaw shifts that cannot improve the current best score. Surviving shifts are always evaluated with the original B = 24 distributions, so the hierarchy changes the order and amount of computation rather than the final similarity definition or yaw resolution. The exhaustive and hierarchical implementations agree on all 13,459 evaluation queries. Table VIII shows compute latency falling from 228.55 to 174.27 ms (1.312×) and mean DB-resident memory from 504.85 to 188.50 MiB; a separate 4,000-query audit preserves every top-1 place and yaw decision. For context, conventional SC averages 18.73 ms/query under the same Wild-Places K = 25, 500- query-per-database clean benchmark protocol. The optimized TNVD-SC is therefore about 9.3× more expensive than SC but still processes 5.74 queries/s on the measured CPU, which supports periodic loop-closure or relocalization queries without requiring execution at every LiDAR frame.

## G. Residual Failure Analysis and Spatial Recovery

On the strict Wild-Places cohort, 98.73% of queries retain a positive in the top-25. Thus, after the support key has formed the candidate set, the dominant residual problem is no longer candidate omission: 85.1% of the remaining failures

![](images/23e9cdb94021c03cc78f08af58d212166f85cca6adb7efd7a8d5af6dd3564204.jpg)  
Fig. 4. Long-range TNVD-SC alias. A 159.0 m false candidate outranks the 1.69 m GT; regions A/B explain 95.8% of the net similarity gap. The lower Fig. 4. Long-range TNVD-SC alias. A 159.0 m false candidate outranks t row collapses the 24-bin TNVD tensor into three height bands (0–6, 6–12, and 12–18 m) for display only; matching uses all B = 24 bins. row collapses the 24-bin TNVD tensor into three height bands (0–6, 6–12,  
are reranking errors. Moreover, 52.6% of those failures select a false location more than 20 m away, showing that the difficult cases are often genuine long-range aliases rather than small threshold-boundary errors. This separation is important because simply increasing K cannot resolve a false candidate that already receives a higher fine-matching score than the correct place.

Figure 4 illustrates such a case. A candidate 159.0 m from the query outranks the 1.69 m ground-truth match because two spatial regions with similar terrain-relative vertical layouts contribute most of the net similarity advantage. Regions A/B account for 95.8% of the score gap, indicating a localized structural alias rather than diffuse descriptor noise. TNVD-SC therefore removes many max-height ambiguities but does not make distinct forest locations uniquely identifiable when their internal vertical organization is itself similar.

The complementary behavior is shown in Fig. 5. Four representative SC failures lie 113.9–631.4 m from the query, yet TNVD-SC recovers matches within 1.1–4.6 m. In these examples, distant places produce similar upper envelopes under scalar max-height pooling, whereas terrain-relative vertical distributions preserve differences in lower and midlevel vegetation occupancy. The qualitative cases therefore mirror the controlled ablations: the main gain comes from preserving within-cell vertical mass for reranking, not from widening the candidate search.

## VI. DISCUSSION

The controlled experiments separate candidate recovery from fine discrimination. The support key nearly saturates Top-25 positive coverage, yet conventional SC matching still loses many retrieved positives; terrain normalization is likewise most effective when the downstream cell representation preserves vertical probability mass. The gain therefore follows the diagnosed fine-matching bottleneck rather than retrieval breadth or ground normalization alone.

Residual failures are mostly reranking ambiguities, and Fig. 4 shows that similar vertical layouts can still outrank the correct place. TreeLoc++ remains stronger in mean Wild Places MR/MF1/AUC and Oxford R@1, while TNVD-SC wins some Oxford sequences, supporting complementary representations. We leave confidence-aware structural routing outside the present method and use NCLT/Oxford to test the frozen TNVD representation directly.

First-bin saturation remains a limitation: the zero-clamped first bin carries 29.9%/31.4% mean occupied-cell probability mass on Venman/Karawatha. A matcher-only control that removes this bin while retaining the frozen 480-D retrieval key improves strict 24-pair mean R@1 by 2.43 points; because this refinement was evaluated only on Wild-Places, we retain the all-bin configuration for generalization experiments.

## VII. CONCLUSION

We presented TNVD-SC, a training-free SC-compatible system derived from a stage-wise diagnosis of forest aliasing. The results show that the dominant gain comes from preserving terrain-relative within-cell vertical structure for fine matching, while a compact support key is sufficient for retrieval. The frozen method reaches 91.83% mean R@1 on standardized Wild-Places, transfers to NCLT and Oxford Forest, and supports exact count-coded hierarchical matching. The unchanged cross-domain evaluation further indicates that this separation is not tied to the Wild-Places tuning domain.

## ACKNOWLEDGMENT

OpenAI ChatGPT was used to assist with manuscript drafting and language refinement and to generate and iteratively

![](images/7bbbfc1957e312548c199bd6f575f715d21a6ca561ace39e6acd62ade7340b53.jpg)

![](images/47bdb5cc96ae848efca030e03740109acabe3954d735bf976965cc6a73002083.jpg)

![](images/6b6b8ff1927cdbba5a2ef1e882e2763d7ad608a1d4d0479e07cded87a6d0f25e.jpg)

![](images/8615aca2e6522a311f5d23a057a2af4e33e8abe980d55ffd85123af27d1e1da3.jpg)  
Database trajectory-—- Query trajectory☆ QueryO Nearest GT· SC Top-1TNVD-SC Top-1

Fig. 5. SC failures recovered by TNVD-SC. P50/P90 examples use representative pairs nearest to each site’s mean gain; insets show the 5 m GT region. Fig. 5. SC failures recovered by TNVD-SC. P50/P90 examples use repres

refine Fig. 1 from author-specified technical content; the authors reviewed and verified the final text and figure for technical accuracy.

## REFERENCES

[1] J. Knights, K. Vidanapathirana, M. Ramezani, S. Sridharan, C. Fookes, and P. Moghadam, “Wild-places: A large-scale dataset for lidar place recognition in unstructured natural environments,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2023, pp. 11 322–11 328.

[2] G. Kim and A. Kim, “Scan context: Egocentric spatial descriptor for place recognition within 3d point cloud map,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS), 2018, pp. 4802–4809.

[3] G. Kim, S. Choi, and A. Kim, “Scan context++: Structural place recognition robust to rotation and lateral variations in urban environments,” IEEE Trans. Robot., vol. 38, no. 3, pp. 1856–1874, 2022.

[4] Y. Shen, T. Tuna, M. Hutter, C. Cadena, and N. Zheng, “Forestlpr: Lidar place recognition in forests attentioning multiple bev density images,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 6659–6669.

[5] E. Griffiths, M. Haghighat, S. Denman, C. Fookes, and M. Ramezani, “Hotformerloc: Hierarchical octree transformer for versatile lidar place recognition across ground and aerial views,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2025, pp. 6648–6658.

[6] ——, “Hotfloc++: End-to-end hierarchical lidar place recognition, re-ranking, and 6-dof metric localisation in forests,” IEEE Robot. Autom. Lett., vol. 11, no. 7, pp. 7860–7867, 2026.

[7] M. Jung, D. Lee, N. Chebrolu, H. Oh, M. Fallon, and A. Kim, “Treeloc++: Robust 6-dof lidar localization in forests with a compact digital forest inventory,” IEEE Trans. Field Robot., 2026.

[8] T. Joseph, T. Fischer, and M. Milford, “Matched filtering based lidar place recognition for urban and natural environments,” IEEE Robotics and Automation Letters, vol. 10, no. 3, pp. 2566–2573, 2025.

[9] C. Bae, G. R. Cho, J. Bae, S. Park, M. Lee, S. Kim, and J. H. Park, “Cylindrical scan context: A multi-channel descriptor for vertical-structure-aware lidar localization,” Sensors, vol. 25, no. 23, p. 7223, 2025.

[10] D. Xu, J. Liu, J. Hyyppa, Y. Liang, and W. Tao, “A heterogeneous 3d map-based¨ place recognition solution using virtual lidar and a polar grid height coding image descriptor,” ISPRS J. Photogramm. Remote Sens., vol. 183, pp. 1–18, 2022.

[11] J. Lee, B. Lee, and G. Yoo, “Probe: Probabilistic occupancy bev encoding with analytical translation robustness for 3d place recognition,” IEEE Robot. Autom. Lett., vol. 11, no. 8, pp. 9779–9786, 2026.

[12] X. Xu, S. Lu, J. Wu, H. Lu, Q. Zhu, Y. Liao, R. Xiong, and Y. Wang, “Ring++: Roto-translation invariant gram for global localization on a sparse scan map,” IEEE Trans. Robot., vol. 39, no. 6, pp. 4616–4635, 2023.

[13] H. Kim, J. Choi, T. Sim, G. Kim, and Y. Cho, “Narrowing your fov with solid: Spatially organized and lightweight global descriptor for fov-constrained lidar place recognition,” IEEE Robot. Autom. Lett., vol. 9, no. 11, pp. 9645–9652, 2024.

[14] M. A. Uy and G. H. Lee, “Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 4470–4479.

[15] J. Komorowski, “Minkloc3d: Point cloud based large-scale place recognition,” in Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV), 2021, pp. 1790– 1799.

[16] ——, “Improving point cloud based place recognition with ranking-based loss and large batch training,” in Proc. Int. Conf. Pattern Recognit. (ICPR), 2022, pp. 3699–3705.

[17] T.-X. Xu, Y.-C. Guo, Z. Li, G. Yu, Y.-K. Lai, and S.-H. Zhang, “Transloc3d: Point cloud based large-scale place recognition using adaptive receptive fields,” Commun. Inf. Syst., vol. 23, no. 1, pp. 57–83, 2023.

[18] K. Vidanapathirana, M. Ramezani, P. Moghadam, S. Sridharan, and C. Fookes, “Logg3d-net: Locally guided global descriptor learning for 3d place recognition,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA), 2022, pp. 2215–2221.

[19] J. Ma, J. Zhang, J. Xu, R. Ai, W. Gu, and X. Chen, “Overlaptransformer: An efficient and yaw-angle-invariant transformer network for lidar-based place recognition,” IEEE Robot. Autom. Lett., vol. 7, no. 3, pp. 6958–6965, 2022.

[20] J. Ma, X. Chen, J. Xu, and G. Xiong, “Seqot: A spatial–temporal transformer network for place recognition using sequential lidar data,” IEEE Trans. Ind. Electron., vol. 70, no. 8, pp. 8225–8234, 2023.

[21] J. Ma, G. Xiong, J. Xu, and X. Chen, “Cvtnet: A cross-view transformer network for lidar-based place recognition in autonomous driving environments,” IEEE Trans. Ind. Informat., vol. 20, no. 3, pp. 4039–4048, 2024.

[22] L. Luo, S.-Y. Cao, X. Li, J. Xu, R. Ai, Z. Yu, and X. Chen, “Bevplace++: Fast, robust, and lightweight lidar global localization for autonomous ground vehicles,” IEEE Trans. Robot., vol. 41, pp. 4479–4498, 2025.

[23] J. Luo, J. Cheng, Q. Xiang, J. Wu, R. Fan, X. Chen, and X. Tang, “Overlapmamba: A shift state space model for lidar-based place recognition,” IEEE Robot. Autom. Lett., vol. 10, no. 8, pp. 8380–8387, 2025.

[24] H. Lim, M. Oh, and H. Myung, “Patchwork: Concentric zone-based region-wise ground segmentation with ground likelihood estimation using a 3d lidar sensor,” IEEE Robotics and Automation Letters, vol. 6, no. 4, pp. 6458–6465, 2021.

[25] S. Lee, H. Lim, and H. Myung, “Patchwork++: Fast and robust ground segmentation solving partial under-segmentation using 3d point cloud,” in 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 13 276–13 283.

[26] N. Carlevaris-Bianco, A. K. Ushani, and R. M. Eustice, “University of michigan north campus long-term vision and lidar dataset,” Int. J. Robot. Res., vol. 35, no. 9, pp. 1023–1035, 2016.