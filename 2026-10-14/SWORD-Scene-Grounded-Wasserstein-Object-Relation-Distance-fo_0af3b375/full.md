# SWORD: Scene-Grounded Wasserstein Object-Relation Distance for 3D Scene Graph Prediction

U-Chae Jun wjsdbco@sookmyung.ac.kr Sookmyung Women’s University Seoul, South Korea

Yoonhyuk Choi<sup>∗</sup>   
chldbsgur123@sookmyung.ac.kr   
Sookmyung Women’s University Seoul, South Korea

## Abstract

3D scene graph prediction is commonly optimized with object-node and relation-edge classification losses. These objectives encourage exact object and relation label recovery, but they do not directly optimize semantic proximity to the ground-truth scene graph: two wrong graphs can receive comparable penalties even when one is much closer in object roles and relation contexts. A direct graphdistance loss is dificult in general, since graph matching and graph edit distance require correspondence search and edit-cost design. Our key observation is that prior supervised 3D scene graph prediction protocols are conducted on fixed object-node indices and supervised edge slots. This can turn a scene graph comparison into a well-defined aligned graph-distance problem. To this end, we propose SWORD, a Scene-Grounded Wasserstein Object-Relation Distance that builds block-normalized object and relation ground distance metrics from training scene graph annotations through a deterministic base-to-context order, without external semantic priors. SWORD lifts these metrics to Wasserstein distances over aligned node and null-augmented edge marginals, providing semantic graph-distance supervision beyond exact-label penalties while preserving the ground-truth graph as the unique zero-distance solution in aligned marginal probability space. We prove that SWORD is a strict metric on aligned marginal probabilistic scene graphs, preserves scene-graph-grounded substitution ordering, controls additive graph-grounded semantic observables, and yields a surrogate upper-bounding decoded semantic error.

## CCS Concepts

• Computing methodologies → Knowledge representation and reasoning; Scene understanding.

## Keywords

3D Scene Graph, Graph Distance, Wasserstein Distance

Jaeeun Ko   
rhwodms1223@sookmyung.ac.kr   
Sookmyung Women’s University Seoul, South Korea

Jiwoo Kang<sup>∗</sup> jwkang@sookmyung.ac.kr Sookmyung Women’s University Seoul, South Korea

ACM Reference Format:   
U-Chae Jun, Jaeeun Ko, Yoonhyuk Choi, and Jiwoo Kang. 2026. SWORD: Scene-Grounded Wasserstein Object-Relation Distance for 3D Scene Graph Prediction. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/ 3799682.3840733

## 1 Introduction

3D scene graph prediction represents a 3D scene as a graph ofobject nodes and relation edges, enabling object-level reasoning beyond isolated recognition. Since 3D semantic scene graphs were introduced for indoor reconstructions, a broad line of work has improved 3D scene graph prediction through RGB-D fusion, point-cloud reasoning, knowledge-guided modeling, visual-linguistic training, multimodal knowledge accumulation, transformers, heterogeneous graph learning, object-centric representations, and edge-centric relation reasoning [1, 6, 10, 11, 22–24, 33–35, 38]. Despite these advances, the supervision objective is still largely based on local label prediction: object nodes are trained to classify object labels, and relation edges are trained to classify relation labels.

Such exact-label supervision preserves the ground-truth label as the target, but it does not define a semantic error geometry among diferent wrong scene graphs. If the ground-truth object is chair, predicting sofa and predicting wall are both incorrect after hard decoding, even though sofa may be closer to chair in the scene graph context. Likewise, if the ground-truth relation is near, predicting adjacent and predicting an opposite directional relation are both wrong, although their semantic severities can difer. Thus, CE/BCEstyle node-edge classification optimizes exact-label likelihood, but it does not define an explicit scene-graph-grounded distance that orders semantically near and far wrong predictions.

Fig. 1 illustrates this limitation. The reference graph is compared with three predicted graphs. Under conventional exact-label scoring, graphs (A) and (B) receive almost the same penalties, 0.31 and 0.30, while graph (C) receives a penalty of 0.62, approximately twice as large. This ordering suggests that (A) and (B) are similarly close to the reference and that (C) is substantially farther away. However, a scene-graph-level reading suggests a diferent ordering. Graph (A) changes object roles and spatial relations in a way that more strongly disrupts the semantic organization of the reference graph. In contrast, graphs (B) and (C) mainly replace objects or relations with alternatives that remain close in the graph context. The example shows that counting local label mismatches can misrepresent the semantic severity of a 3D scene graph prediction error. A desirable loss should therefore preserve the ground-truth graph as the exact optimum, while assigning larger penalties to semantically disruptive substitutions and smaller penalties to graph-grounded alternatives that remain close to the reference.

A natural alternative is graph similarity, but general graph comparison is ill-posed unless one specifies node correspondence, edge correspondence, edit operations, and edit costs. Classical graph edit distance handles this by searching over node and edge insertions, deletions, and substitutions, but the resulting optimization is combinatorial and dificult to use directly as a training objective [2, 5, 7, 26– 28]. Our key observation is that many 3D scene graph prediction protocols are not arbitrary graph comparison problems: the pre dicted graph and the ground-truth graph share object-node indices and supervised relation edge slots [10, 23, 24, 33, 35]. This fixed correspondence turns graph similarity from a matching-dependent problem into a well-posed aligned node-and-edge distance problem.

To this end, we propose SWORD, a Scene-grounded Wasserstein Object-Relation Distance for 3D scene graph prediction. SWORD constructs object and relation ground metrics from training scene graph annotations, using object labels, relation labels, directed edge slots, endpoint contexts, local graph neighborhoods, relation cooccurrences, and typed-degree profiles. Instead of relying on external semantic priors such as language embeddings, knowledge bases, or vision-language models, SWORD derives its semantic geometry from the training scene graphs themselves. To avoid circularity be tween object and relation semantics, SWORD constructs these costs through a deterministic base-to-context order, from base object and relation metrics to context-aware object and relation metrics. It then uses Wasserstein distances to lift these costs into probability spaces over aligned node and edge marginals [8, 31].

The resulting objective changes the supervision geometry while preserving exact ground-truth optimality. A wrong prediction that is close under the scene-graph-grounded metric can receive a smaller penalty than a semantically distant wrong prediction, but the exact ground-truth graph remains the unique zero-distance solution in probability space. For practical single-label training, we introduce SWORD-Margin, a metric-induced structured log loss that uses SWORD distances as semantic margins. The construction is deterministic once the graph-profile blocks and edge-slot protocol are specified, and it does not require external semantic priors or CE/BCE mixing weights.

The main contributions are summarized as follows. First, we identify fixed correspondence as the structural property that makes semantic graph distance well posed for supervised 3D scene graph prediction, turning graph comparison from a matching-dependent problem into an aligned object-relation distance problem. Second, we construct scene-graph-grounded object and relation metrics from block-normalized training graph profiles, without relying on external semantic priors. Third, we define SWORD as a Wasserstein object-relation graph distance over aligned node and edge marginals, with a null-augmented relation metric for no relation edge slots. Fourth, we introduce SWORD-Margin for practical single-label training and establish that SWORD is a strict metric on aligned marginal probabilistic scene graphs, preserves scenegraph-grounded substitution ordering, controls additive graphgrounded semantic observables, and has the exact ground-truth graph as the unique zero-distance minimizer.

## 2 Related Work

## 2.1 3D Scene Graph Prediction

3D scene graph prediction is a structured scene understanding task within the field of 3D vision [12–14]. It represents a 3D scene as object nodes and relation edges, and has evolved from indoor reconstruction-based scene graphs to RGB-D, point-cloud, transformer, heterogeneous-graph, object-centric, and edge-centric formulations [6, 10, 22–24, 33–35, 38]. Recent extensions further incorporate visual-linguistic supervision, multimodal knowledge, and pretraining strategies [6, 34]. Across this line of work, the standard prediction interface supervises object nodes and relation edges with annotated labels, and training is typically formulated as independent node and edge classification with CE, BCE, or reweighted classification objectives over object and relation vocabularies [10, 22– 24, 33, 35, 38].

These objectives make the ground-truth label optimal, but their label geometry remains exact-match based: they do not specify whether one incorrect object or relation label is semantically closer to the ground truth than another. Consequently, after hard decoding, two wrong labels have the same exact-error status even if one better preserves the scene-graph semantics of the ground-truth scene [6, 22, 24, 33, 34]. SWORD is orthogonal to architectural advances: it exploits the fixed-correspondence node-edge interface to change the supervision geometry from exact-label matching to a scene-graph-grounded object-relation distance built from blocknormalized training graph profiles.

## 2.2 Graph Distance and Matching

General graph similarity is dificult because node correspondence, edge correspondence, attribute substitution, and topology changes must be handled jointly [5, 7, 27, 28]. Graph edit distance provides a classical dissimilarity measure based on node and edge insertion, deletion, and substitution, but exact or approximate computation typically involves combinatorial matching procedures such as editpath search or bipartite matching [2, 7, 26].

The key observation behind SWORD is that this ambiguity disappears in the fixed-correspondence interface used by many 3D scene graph prediction protocols, where predicted and groundtruth graphs share object-node indices and supervised edge slots [10, 23, 24, 33, 35]. This turns graph comparison into a well-posed problem over aligned node and edge marginals, rather than an editpath or matching problem over arbitrary graph correspondences. Thus, SWORD defines a metric on aligned marginal graph tuples induced by the supervised 3D scene graph prediction interface, not on unaligned graphs.

## 2.3 Wasserstein Ground Metrics

The 1-Wasserstein distance compares probability measures through an explicit ground cost, making it suitable when diferent label substitutions should have diferent semantic severities [8, 31]. This difers from standard classification losses, which mainly score the ground-truth label probability without specifying object-relation substitution costs. SWORD uses Wasserstein distances to lift object and relation ground metrics into node-level, edge-level, and graph level distances over aligned scene graph marginals. The resulting Wasserstein graph distance provides the metric definition, while the cost-augmented log-sum-exp objective used for single-label training is a metric-induced surrogate rather than a separate graph metric.

![](images/42918484b8b5332a460d8b0c32333f2b1f96264e5f54eecadf863b041566b421.jpg)  
Figure 1: Existing CE/BCE-style losses mainly reflect local object and relation label mismatches, making (A) and (B) appear similarly close to the reference and (C) about twice as far. A scene-graph-level interpretation suggests that (A) is more disruptive, whereas (B) and (C) remain semantically closer to the reference. � previews the graph-grounded ordering by our method.

Semantic information can also be injected through distributional vector-space semantics, language priors, knowledge bases, vision-language models, or open-vocabulary contrastive learning [3, 4, 6, 16, 21, 30, 34]. SWORD instead constructs its ground metrics from training scene graph annotations themselves, using objectrelation contexts, endpoint statistics, local neighborhoods, relation co-occurrences, and typed-degree profiles [23, 24, 33, 35, 38]. This makes the induced semantics dataset-internal and scene-graphgrounded rather than externally imposed. To avoid circularity between object semantics and relation semantics, SWORD constructs these costs through a deterministic base-to-context order, first obtaining base object and relation metrics and then refining them with context-aware object and relation profiles.

## 3 Method

We consider fixed-correspondence 3D scene graph prediction, where object nodes and directed relation edges are supervised by annotated training scene graphs [6, 10, 22–24, 33–35]. Existing CE and BCE objectives penalize node and edge label errors independently, but do not define a graph-level semantic distance to the groundtruth scene graph. SWORD addresses this gap by constructing scene-graph-grounded object and relation metrics from training graph statistics, and lifting them to a Wasserstein distance between aligned probabilistic scene graphs.

## 3.1 3D Scene Graph Estimation Problem

Let the object and relation label spaces be C and R, and let ∅ denote the null relation:

$$
C = \{ 1 , . . . , C \} , \qquad \mathcal { R } = \{ 1 , . . . , K \} , \qquad \mathcal { R } _ { \emptyset } = \mathcal { R } \cup \{ \emptyset \} .\tag{1}
$$

For a supervised scene, the predicted graph and the ground-truth graph share the same object-node indices and directed edge-slot

indices:

$$
V = \{ 1 , . . . , n \} , \qquad \mathcal { A } \subseteq \{ ( i , j ) : i , j \in V , i \neq j \} .\tag{2}
$$

The set A is determined by the benchmark protocol or the prediction architecture, and is not required to contain all ordered pairs. This fixed-correspondence structure is a key observation: general graph comparison requires node correspondences and edit costs, whereas supervised 3D scene graph prediction already provides aligned object-node and edge-slot coordinates. SWORD therefore formulates scene graph comparison as a well-posed metric problem on aligned marginal graph tuples, rather than as a combinatorial graph edit distance or graph matching problem [2, 5, 7, 26, 27].

For node � and edge slot (�, �), the model outputs

$$
p _ { i } \in \Delta ( C ) , \qquad y _ { i } ^ { * } \in C ,\tag{3}
$$

$$
q _ { i j } \in \Delta ( \mathcal { R } _ { \varnothing } ) , \qquad r _ { i j } ^ { \ast } \in \mathcal { R } _ { \varnothing } \quad \mathrm { f o r s i n g l e - l a b e l r e l a t i o n s } ,\tag{4}
$$

$$
\begin{array} { r } { s _ { i j } ( r ) \in [ 0 , 1 ] , \qquad S _ { i j } ^ { * } \subseteq \mathcal { R } \quad \mathrm { f o r ~ m u l t i \mathrm { - } l a b e l ~ r e l a t i o n s . } } \end{array}\tag{5}
$$

An aligned marginal probabilistic scene graph is represented as

$$
G = \left( \{ p _ { i } ^ { G } \} _ { i \in V } , \{ \mu _ { i j } ^ { G } \} _ { ( i , j ) \in \mathcal { A } } \right) , \qquad G ^ { * } = \left( \{ \delta _ { y _ { i } ^ { * } } \} _ { i \in V } , \{ \mu _ { i j } ^ { \mathrm { g t } } \} _ { ( i , j ) \in \mathcal { A } } \right) .\tag{6}
$$

SWORD is a metric on aligned marginal graph tuples of the form in (6). It is not a metric over arbitrary unaligned graphs or unspecified full joint graph distributions.

For finite distributions over a token space, SWORD uses total variation when no lower-level ground metric is defined, and Wasserstein distance otherwise:

$$
d _ { \mathrm { T V } } ( P , Q ) = \frac { 1 } { 2 } \sum _ { x \in \cal { X } } | P ( x ) - Q ( x ) | ,\tag{7}
$$

$$
W _ { 1 } ( \mu , \nu ; d ) = \operatorname* { m i n } _ { \pi \in \Pi ( \mu , \nu ) } \sum _ { x , y \in \mathcal { X } } \pi ( x , y ) d ( x , y ) .\tag{8}
$$

When � is a metric, $W _ { 1 } ( \cdot , \cdot ; d )$ is a metric on probability measures over X [8, 31].

## 3.2 Scene-Graph-Grounded Ground Metrics

SWORD constructs object and relation ground metrics only from training scene graph annotations: object labels, relation labels, directed incidence, endpoint contexts, local neighborhoods, typed degrees, and relation co-occurrence. It does not use raw 3D coordinates, geometry, shape descriptors, point-cloud features, language priors, or vision-language priors. Thus, the induced semantics are dataset-internal and scene-graph-grounded.

Object and relation semantics are mutually dependent in a scene graph. Object labels are characterized by their relation and neighbor contexts, while relation labels are characterized by endpoint objects and local relation contexts. To avoid a circular definition, SWORD uses an acyclic base-to-context construction:

$$
d _ { O } ^ { \mathrm { b a s e } } \quad \longrightarrow \quad d _ { R } ^ { \mathrm { b a s e } } \quad \longrightarrow \quad d _ { O } \quad \longrightarrow \quad d _ { R } .\tag{9}
$$

Empirical graph profiles. For a graph-profile channel � with finite token space ${ \bar { X } } _ { m } ,$ let $\mathcal { T } _ { \ell , m }$ be the set of training occurrences associated with label ℓ, and let � (� $\phi _ { m } ( \alpha )$ be the token extracted from occurrence �. A channel-specific empty token $\mathbb { Q } _ { m }$ is used when no occurrence is observed. The empirical profile is

$$
\widehat { P } _ { \ell , m } ( x ) = \left\{ \begin{array} { l l } { \frac { 1 } { | { \mathcal I } _ { \ell , m } | } \sum _ { \alpha \in { \mathcal I } _ { \ell , m } } { \mathbf 1 } [ \phi _ { m } ( \alpha ) = x ] , } & { | { \mathcal I } _ { \ell , m } | > 0 , } \\ { { \mathbf 1 } [ x = { \mathcal O } _ { m } ] , } & { | { \mathcal I } _ { \ell , m } | = 0 . } \end{array} \right.\tag{10}
$$

Channels are grouped into semantically homogeneous blocks so that no profile family dominates merely by containing more chan nels. For block family $\mathcal { B } _ { t }$ at construction stage �, define

$$
\mathrm { B A } _ { \mathcal { B } _ { t } } \left[ \left\{ A _ { m } \right\} _ { m } \right] ( \ell , \ell ^ { \prime } ) = \frac { 1 } { | \mathcal { B } _ { t } | } \sum _ { B \in \mathcal { B } _ { t } } \frac { 1 } { | \mathcal { M } _ { B } | } \sum _ { m \in \mathcal { M } _ { B } } A _ { m } ( \ell , \ell ^ { \prime } ) .\tag{11}
$$

The object blocks include neighbor-object, incoming-relation, outgoingrelation, relation-neighbor, and typed-degree profiles. The relation blocks include subject-class, object-class, subject-object pair, reverse-relation, and local relation co-occurrence profiles. These block families are fixed before training and are part of the method specification.

When a Wasserstein channel contains $\varnothing _ { m } ,$ a non-empty ground metric $d _ { m } ^ { + }$ with diameter at most one is extended as

$$
\begin{array} { r } { d _ { m } ^ { \mathcal { O } } ( x , y ) = \left\{ \begin{array} { l l } { d _ { m } ^ { + } ( x , y ) , } & { x , y \neq \emptyset _ { m } , } \\ { 1 , } & { x \neq \emptyset _ { m } , \ y = \emptyset _ { m } , } \\ { 1 , } & { x = \emptyset _ { m } , \ y \neq \emptyset _ { m } , } \\ { 0 , } & { x = y = \emptyset _ { m } . } \end{array} \right. } \end{array}\tag{12}
$$

Strict correction. Empirical profile distances may be pseudometrics because distinct labels can have identical graph profiles. For a finite label space L and raw pseudometric $d _ { L } ^ { 0 } ,$ , define

$$
\Delta _ { L } = \operatorname* { m a x } _ { u , v \in \mathcal { L } } d _ { L } ^ { 0 } ( u , v ) , \qquad \bar { d } _ { L } ^ { 0 } ( u , v ) = \left\{ \begin{array} { l l } { d _ { L } ^ { 0 } ( u , v ) / \Delta _ { L } , } & { \Delta _ { L } > 0 , } \\ { 0 , } & { \Delta _ { L } = 0 , } \end{array} \right.
$$

$$
{ \mathcal P } _ { L } = \{ \bar { d } _ { L } ^ { 0 } ( u , v ) : u \neq v , \bar { d } _ { L } ^ { 0 } ( u , v ) > 0 \} , \qquad \eta _ { L } = \left\{ \begin{array} { l l } { \frac { 1 } { 2 } \operatorname* { m i n } \mathcal { P } _ { L } , } & { \mathcal { P } _ { L } \neq 0 , } \\ { 1 , } & { \mathcal { P } _ { L } = 0 , } \end{array} \right.
$$

$$
\mathrm { S C } _ { \mathcal { L } } [ d _ { L } ^ { 0 } ] ( u , v ) = \frac { \operatorname* { m a x } \big ( \bar { d } _ { L } ^ { 0 } ( u , v ) , \eta _ { L } 1 [ u \neq v ] \big ) } { \operatorname* { m a x } _ { a , b \in \mathcal { L } } \operatorname* { m a x } \Big ( \bar { d } _ { L } ^ { 0 } ( a , b ) , \eta _ { L } 1 [ a \neq b ] \Big ) } .\tag{13}
$$

The scale $\eta _ { L }$ is determined by the training-set distance matrix and is not tuned on validation data. It separates only label pairs indistinguishable under the chosen profiles, without changing the ordering of pairs already separated by nonzero graph-grounded distance. Base metrics. The base object metric compares exact graph-token profiles by total variation:

$$
A _ { m } ^ { { \cal O } , \mathrm { b a s e } } ( a , b ) = d _ { \mathrm { T V } } \left( \widehat { P } _ { a , m } , \widehat { P } _ { b , m } \right) ,\tag{14}
$$

$$
\begin{array} { r } { d _ { O , \mathrm { b a s e } } ^ { 0 } ( a , b ) = \mathrm { B A } _ { \mathcal { B } _ { O } ^ { \mathrm { b a s e } } } \left[ \{ A _ { m } ^ { O , \mathrm { b a s e } } \} _ { m } \right] ( a , b ) , \qquad d _ { O } ^ { \mathrm { b a s e } } = \mathrm { S C } _ { C } [ d _ { O , \mathrm { b a s e } } ^ { 0 } ] . } \end{array}
$$

Using $d _ { O } ^ { \mathrm { b a s e } }$ , define a base endpoint-pair metric and the base relation metric:

$$
d _ { \mathrm { p a i r } } ^ { \mathrm { b a s e } } \big ( ( a , b ) , ( a ^ { \prime } , b ^ { \prime } ) \big ) = \frac { d _ { O } ^ { \mathrm { b a s e } } ( a , a ^ { \prime } ) + d _ { O } ^ { \mathrm { b a s e } } ( b , b ^ { \prime } ) } { 2 } ,\tag{15}
$$

$$
A _ { m } ^ { R , \mathrm { b a s e } } ( r , s ) = D _ { m } ^ { R , \mathrm { b a s e } } \left( \widehat { P } _ { r , m } , \widehat { P } _ { s , m } \right) ,\tag{16}
$$

$$
\begin{array} { r } { d _ { R , \mathrm { b a s e } } ^ { 0 } ( r , s ) = \mathrm { B A } _ { \mathcal { B } _ { R } ^ { \mathrm { b a s e } } } \left[ \{ A _ { m } ^ { R , \mathrm { b a s e } } \} _ { m } \right] ( r , s ) , ~ d _ { R } ^ { \mathrm { b a s e } } = \mathrm { S C } _ { \mathcal { R } } [ d _ { R , \mathrm { b a s e } } ^ { 0 } ] . } \end{array}
$$

Here subject and object channels use $W _ { 1 } ( \cdot , \cdot ; d _ { O } ^ { \mathrm { b a s e } } )$ , endpoint-pair channels use $W _ { 1 } ( \cdot , \cdot ; d _ { \mathrm { p a i r } } ^ { \mathrm { b a s e } } )$ , and exact relation-token channels use $d _ { \mathrm { T V } }$

Context-aware metrics. The context-aware object metric compares object profiles using the base object and relation metrics as ground costs. First define

$$
d _ { \mathrm { r e l - n b r } } ^ { \mathrm { b a s e } } \big ( ( r , c ) , ( r ^ { \prime } , c ^ { \prime } ) \big ) = \frac { d _ { R } ^ { \mathrm { b a s e } } ( r , r ^ { \prime } ) + d _ { O } ^ { \mathrm { b a s e } } ( c , c ^ { \prime } ) } { 2 } .\tag{17}
$$

The context-aware object metric is

$$
\begin{array} { r l } & { A _ { m } ^ { O , \mathrm { c t x } } ( a , b ) = D _ { m } ^ { O } \left( \widehat { P } _ { a , m } , \widehat { P } _ { b , m } \right) , } \\ & { \qquad d _ { O } ^ { 0 } ( a , b ) = \mathrm { B A } _ { \mathcal { B } _ { O } ^ { \mathrm { c t x } } } \left[ \{ A _ { m } ^ { O , \mathrm { c t x } } \} _ { m } \right] ( a , b ) , \qquad d _ { O } = \mathrm { S C } _ { C } [ d _ { O } ^ { 0 } ] . } \end{array}\tag{18}
$$

In (18), neighbor-object channels use $W _ { 1 } ( \cdot , \cdot ; d _ { O } ^ { \mathrm { b a s e } } )$ , relation-label channels use $W _ { 1 } ( \cdot , \cdot ; d _ { R } ^ { \mathrm { b a s e } } )$ , relation-neighbor channels use $W _ { 1 } ( \cdot , \cdot ; d _ { \mathrm { r e l - n b r } } ^ { \mathrm { b a s e } } ) _ { \mathrm { ; } }$ and typed-degree channels use $d _ { \mathrm { T V } }$

Finally, the context-aware relation metric uses the object metric $d _ { O }$ for endpoint comparison and $d _ { R } ^ { \mathrm { b a s e } }$ for relation-token comparison:

$$
\begin{array} { r l } & { d _ { \mathrm { p a i r } } \big ( ( a , b ) , ( a ^ { \prime } , b ^ { \prime } ) \big ) = \frac { d _ { O } \left( a , a ^ { \prime } \right) + d _ { O } ( b , b ^ { \prime } ) } { 2 } , } \\ & { A _ { m } ^ { R , \mathrm { c t x } } ( r , s ) = D _ { m } ^ { R } \left( \widehat { P } _ { r , m } , \widehat { P } _ { s , m } \right) , } \\ & { d _ { R } ^ { 0 } ( r , s ) = \mathrm { B A } _ { \mathcal { B } _ { R } ^ { \mathrm { c t x } } } \left[ \{ A _ { m } ^ { R , \mathrm { c t x } } \} _ { m } \right] ( r , s ) , \qquad d _ { R } = \mathrm { S C } _ { \mathcal { R } } [ d _ { R } ^ { 0 } ] . } \end{array}\tag{19}
$$

(20)

Subject and object channels use $W _ { 1 } ( \cdot , \cdot ; d _ { O } )$ , endpoint-pair channels use $W _ { 1 } ( \cdot , \cdot ; d _ { \mathrm { p a i r } } )$ , and reverse-relation and co-occurrence channels use $W _ { 1 } ( \cdot , \cdot ; d _ { R } ^ { \mathrm { b a s e } } )$ . The construction is deterministic and noniterative.

Null augmentation. Since $d _ { R }$ has normalized diameter one, we extend it to $\mathcal { R } _ { \varnothing }$ by

$$
d _ { R } ^ { 0 } ( x , y ) = \left\{ \begin{array} { l l } { d _ { R } ( x , y ) , } & { x , y \in \mathcal { R } , } \\ { 1 , } & { x \in \mathcal { R } , y = 0 , } \\ { 1 , } & { x = 0 , y \in \mathcal { R } , } \\ { 0 , } & { x = y = 0 . } \end{array} \right.\tag{21}
$$

The null cost is tied to the normalized diameter of $d _ { R }$ and is not selected by validation as an insertion or deletion cost.

## 3.3 SWORD Distance and Training Objectives

For a node �, the SWORD node atom distance to the ground truth is

$$
\ell _ { i } ^ { V } = W _ { 1 } ( p _ { i } , \delta _ { y _ { i } ^ { * } } ; d _ { O } ) = \sum _ { c \in C } \bar { p } _ { i } ( c ) d _ { O } ( c , y _ { i } ^ { * } ) .\tag{22}
$$

For a single-label edge slot $( i , j )$ , the edge atom distance is

$$
\ell _ { i j } ^ { E } = W _ { 1 } ( q _ { i j } , \delta _ { r _ { i j } ^ { * } } ; d _ { R } ^ { 0 } ) = \sum _ { r \in \mathcal { R } _ { \varnothing } } q _ { i j } ( r ) d _ { R } ^ { \varnothing } ( r , r _ { i j } ^ { * } ) .\tag{23}
$$

For multi-label relation prediction, the sigmoid vector and groundtruth relation set are converted into probability measures on $\mathcal { R } _ { \varnothing } \colon$

$$
\mu _ { i j } ^ { \mathrm { p r e d } } = \frac { 1 } { K } \sum _ { r \in \mathcal { R } } s _ { i j } ( r ) \delta _ { r } + \left( 1 - \frac { 1 } { K } \sum _ { r \in \mathcal { R } } s _ { i j } ( r ) \right) \delta _ { \emptyset } ,\tag{24}
$$

$$
\mu _ { i j } ^ { \mathrm { g t } } = \frac { 1 } { K } \sum _ { r \in S _ { i j } ^ { * } } \delta _ { r } + \left( 1 - \frac { | S _ { i j } ^ { * } | } { K } \right) \delta _ { 0 } ,\tag{25}
$$

$$
\begin{array} { r } { \ell _ { i j } ^ { E , \mathrm { m u l t i } } = K \cdot W _ { 1 } \left( \mu _ { i j } ^ { \mathrm { p r e d } } , \mu _ { i j } ^ { \mathrm { g t } } ; d _ { R } ^ { 0 } \right) . } \end{array}\tag{26}
$$

The factor � is fixed by the relation vocabulary size and compensates for the $1 / K$ mass assigned to each real relation slot. For the graph-level notation below, $\mu _ { i j }$ denotes the edge marginal used by SWORD. In the single-label setting, $\mu _ { i j } ^ { \mathrm { p r e d } } = q _ { i j }$ and $\mu _ { i j } ^ { \mathrm { g t } } = \delta _ { r _ { i j } ^ { * } }$ . In the multi-label setting, $\mu _ { i j } ^ { \mathrm { p r e d } }$ and $\mu _ { i j } ^ { \mathrm { g t } }$ are the null-augmented measures defined in Eqs. (24) and (25).

For two aligned marginal scene graphs � and �, define

$$
\beta = \left\{ { 1 , \atop N } \right. \quad \mathrm { s i n g l e - l a b e l ~ r e l a t i o n ~ s e t t i n g } ,\tag{27}
$$

$$
D _ { \beta } ( G , H ) = \frac { \sum _ { i \in V } W _ { 1 } ( \boldsymbol { \rho } _ { i } ^ { G } , \boldsymbol { \rho } _ { i } ^ { H } ; d _ { O } ) + \beta \sum _ { ( i , j ) \in \mathcal { R } } W _ { 1 } ( \boldsymbol { \mu } _ { i j } ^ { G } , \boldsymbol { \mu } _ { i j } ^ { H } ; d _ { R } ^ { \emptyset } ) } { | V | + \beta | \mathcal { A } | } .\tag{28}
$$

The pure SWORD loss is

$$
\mathcal { L } _ { \mathrm { S W O R D } } ( \hat { G } , G ^ { * } ) = D _ { \beta } ( \hat { G } , G ^ { * } ) .\tag{29}
$$

For single-label logit training, Wasserstein-to-Dirac losses are bounded linear functions of predicted probabilities and may give weak gradients for confident wrong predictions. We therefore use a costaugmented log-sum-exp surrogate:

$$
\ell _ { i } ^ { V , \mathrm { m a r } } = \log \sum _ { c \in C } \exp \left( z _ { i } ( c ) + d _ { O } ( c , y _ { i } ^ { \ast } ) \right) - z _ { i } ( y _ { i } ^ { \ast } ) ,\tag{30}
$$

$$
\ell _ { i j } ^ { E , \mathrm { m a r } } = \log \sum _ { \boldsymbol { r } \in \mathcal { R } _ { 0 } } \exp \left( u _ { i j } ( \boldsymbol { r } ) + d _ { R } ^ { \varnothing } ( \boldsymbol { r } , r _ { i j } ^ { * } ) \right) - u _ { i j } ( r _ { i j } ^ { * } ) ,\tag{31}
$$

$$
\mathcal { L } _ { \mathrm { S W O R D - M a r g i n } } = \frac { \sum _ { i \in V } \ell _ { i } ^ { V , \mathrm { m a r } } + \sum _ { ( i , j ) \in \mathcal { A } } \ell _ { i j } ^ { E , \mathrm { m a r } } } { \left| V \right| + \left| \mathcal { A } \right| } .\tag{32}
$$

SWORD-Margin is not itself a metric on probabilistic scene graphs; it is a metric-induced structured log loss whose semantic margins are exactly the SWORD ground distances. This cost-augmented form is related to large-margin and softmax-margin structured prediction, but the margin values here are induced by scene-graphgrounded object and relation metrics rather than by hand-designed task costs [9, 29]. For multi-label relations, we use

$$
\mathcal { L } _ { \mathrm { S W O R D } } ^ { \mathrm { m u l t i } } = \frac { \sum _ { i \in V } \ell _ { i } ^ { V , \mathrm { m a r } } + \sum _ { ( i , j ) \in \mathcal { A } } \ell _ { i j } ^ { E , \mathrm { m u l t i } } } { \left| V \right| + K \left| \mathcal { A } \right| } .\tag{33}
$$

In Eq. (33), the edge term is the multi-label Wasserstein relation loss, while the object term uses the same SWORD-Margin surrogate as in Eq. (30), because object prediction remains single-label in both relation protocols.

## 3.4 Theoretical Properties

Proposition 1. Ground distances are strict metrics. The distances $d _ { O }$ and $d _ { R }$ in (18) and (20) are strict metrics on C and R, respectively.

Proof. Total variation is a metric on finite categorical distributions, and Wasserstein distance is a metric when its ground cost is a metric. The empty-token extension in (12) is a metric because the non-empty diameter is at most one and the empty token is placed at distance one from all non-empty tokens. Therefore, all profile distances used in the acyclic order of (9) induce pseudometrics on label spaces, and the block-normalized averages in (11) preserve pseudometricity as positive weighted sums. The strict correction in (13) takes the pointwise maximum with a scaled discrete metric, which preserves the triangle inequality:

$$
\begin{array} { r l } & { \operatorname* { m a x } ( d ( x , z ) , e ( x , z ) ) \leq \operatorname* { m a x } ( d ( x , y ) + d ( y , z ) , e ( x , y ) + e ( y , z ) ) } \\ & { \quad \quad \quad \quad \leq \operatorname* { m a x } ( d ( x , y ) , e ( x , y ) ) + \operatorname* { m a x } ( d ( y , z ) , e ( y , z ) ) . } \end{array}\tag{34}
$$

It also gives positive separation to all distinct labels, and the final normalization preserves metric properties. Hence $d _ { O }$ and $d _ { R }$ are strict metrics. □

Proposition 2. The null-augmented relation distance is a metric. The distance $d _ { R } ^ { 0 }$ in (21) is a metric on $\mathcal { R } _ { \varnothing }$

Proof. For real relations, the claim follows from Proposition 1. For any $r , s \in { \mathcal { R } }$ , the only nontrivial triangle involving ∅ is

$$
d _ { R } ^ { 0 } ( r , s ) = d _ { R } ( r , s ) \leq 1 \leq d _ { R } ^ { 0 } ( r , \theta ) + d _ { R } ^ { 0 } ( \theta , s ) .\tag{35}
$$

All other cases are immediate from (21).

□

Theorem 1. SWORD is a strict metric on aligned marginal scene graphs. For fixed � and ${ \mathcal { A } } , D _ { \beta }$ in (28) is a strict metric on aligned marginal graph tuples.

Proof. By Propositions 1 and 2, �<sub>�</sub> and $d _ { R } ^ { 0 }$ are strict metrics. Therefore, each Wasserstein term in (28) is a metric on the corresponding probability simplex. A positive weighted sum of metrics followed by positive normalization is a metric. If ${ \bf \dot { \cal D } } _ { \beta } ( G , H ) = 0 .$ , every aligned node marginal and every aligned edge marginal coincide:

$$
\begin{array} { r } { \boldsymbol { p } _ { i } ^ { G } = \boldsymbol { p } _ { i } ^ { H } \quad \forall i \in V , \qquad \boldsymbol { \mu } _ { i j } ^ { G } = \boldsymbol { \mu } _ { i j } ^ { H } \quad \forall ( i , j ) \in \mathcal { A } . } \end{array}\tag{36}
$$

Under fixed correspondence, this is equality of aligned marginal scene graph tuples. □

Theorem 2. SWORD preserves scene-graph-grounded substitution ordering. Let $G ^ { * }$ be the ground-truth graph. Suppose hard graphs $H _ { a }$ and $\bar { H _ { b } }$ difer from $G ^ { * }$ only at node �, where they assign object labels � and �, respectively. If

$$
d _ { O } ( a , y _ { i } ^ { * } ) < d _ { O } ( b , y _ { i } ^ { * } ) ,\tag{37}
$$

then

$$
{ \cal D } _ { \beta } ( H _ { a } , G ^ { * } ) < { \cal D } _ { \beta } ( H _ { b } , G ^ { * } ) .\tag{38}
$$

Proof. All graph coordinates cancel except node �:

$$
D _ { \beta } ( H _ { a } , G ^ { * } ) - D _ { \beta } ( H _ { b } , G ^ { * } ) = \frac { d _ { O } ( a , y _ { i } ^ { * } ) - d _ { O } ( b , y _ { i } ^ { * } ) } { | V | + \beta | \mathcal { A } | } .\tag{39}
$$

The right-hand side is negative by (37). The relation case is identical with $\bar { d } _ { R } ^ { 0 }$ replacing $d _ { O }$ □

Theorem 3. SWORD controls graph-grounded semantic observables. Let $\varphi _ { i } : C \to \mathbb { R }$ be 1-Lipschitz with respect to $d _ { O } ,$ , and let $\psi _ { i j } : \mathcal { R } _ { 0 } $ R be 1-Lipschitz with respect to $d _ { R } ^ { 0 } .$ . Define

$$
\Psi ( G ) = \frac { \sum _ { i \in V } \mathbb { E } _ { c \sim p _ { i } ^ { G } } [ \varphi _ { i } ( c ) ] + \beta \sum _ { ( i , j ) \in \mathcal { R } } \mathbb { E } _ { r \sim \mu _ { i j } ^ { G } } [ \psi _ { i j } ( r ) ] } { | V | + \beta | \mathcal { R } | } .\tag{40}
$$

Then, for any aligned marginal graphs � and �,

$$
| \Psi ( G ) - \Psi ( H ) | \leq D _ { \beta } ( G , H ) .\tag{41}
$$

Proof. By the Kantorovich-Rubinstein dual characterization of $W _ { 1 }$ the expectation diference of any 1-Lipschitz observable is bounded by the corresponding Wasserstein distance. Applying this coordinatewise to all nodes and edges, summing, and normalizing gives (41). □

Corollary 1. The ground-truth graph is the unique zero-distance minimizer. For the pure SWORD loss,

$$
{ \mathcal { L } } _ { \mathrm { S W O R D } } ( { \hat { G } } , G ^ { * } ) = 0 \quad \iff \quad { \hat { G } } = G ^ { * }\tag{42}
$$

in the aligned marginal probability space. With finite softmax or sigmoid logits, exact zero may only be approached in the infinitelogit limit, as in CE and BCE.

Proposition 3. SWORD-Margin upper-bounds decoded semantic error. Let

$$
\hat { y } _ { i } = \underset { c \in C } { \arg \operatorname* { m a x } } z _ { i } ( c ) , \qquad \hat { r } _ { i j } = \underset { r \in \mathcal { R } _ { \varnothing } } { \arg \operatorname* { m a x } } u _ { i j } ( r ) .\tag{43}
$$

Then

$$
\ell _ { i } ^ { V , \mathrm { m a r } } \geq d _ { O } \big ( \hat { y } _ { i } , y _ { i } ^ { * } \big ) , \qquad \ell _ { i j } ^ { E , \mathrm { m a r } } \geq d _ { R } ^ { \varnothing } \big ( \hat { r } _ { i j } , r _ { i j } ^ { * } \big ) .\tag{44}
$$

Proof. The log-sum-exp term is at least the decoded-class term, and the decoded logit is no smaller than the ground-truth logit. Applying this to (30) and (31) gives (44). □

SWORD does not claim to recover universal human commonsense similarity. Its semantics are dataset-internal and graph-grounded. Theorems 2 and 3 show that the ordering and Lipschitz stability induced by the constructed object and relation metrics are preserved at the aligned graph-distance level for single-coordinate substitutions and additive aligned node-edge observables.

## 3.5 Implementation and Complexity

SWORD separates ofline metric construction from online training. Ofline, it extracts graph profiles from the training set, computes $d _ { O } ^ { \mathrm { b a s e } } , d _ { R } ^ { \mathrm { b a s e } } , d _ { O } , d _ { R } ,$ and $d _ { R } ^ { 0 } .$ then stores the object and relation cost matrices. Online, the loss uses only cost-matrix lookup and standard probability or log-sum-exp operations.

Let $M _ { O }$ and $M _ { R }$ denote the total numbers of object and relation graph-profile channels across all blocks, and let $m \ : = \ : | { \mathcal { A } } |$ . The one-time precomputation and online costs are

$$
T _ { \mathrm { p r e } } = O ( C ^ { 2 } M _ { O } + K ^ { 2 } M _ { R } ) , \qquad M _ { \mathrm { s t o r e } } = O ( C ^ { 2 } + ( K + 1 ) ^ { 2 } ) ,\tag{45}
$$

$$
T _ { \mathrm { s i n g l e } } = O ( n C + m K ) , \qquad T _ { \mathrm { m u l t i } } = O ( n C + m K ^ { 3 } ) .\tag{46}
$$

Thus, in the single-label setting, SWORD-Margin has the same asymptotic order as standard softmax node and relation classification, with additional fixed semantic cost lookup.

Once the graph-profile block families and benchmark edge-slot protocol are specified before validation, SWORD introduces no validation-tuned loss-side scalar weight, relation insertion cost, relation deletion cost, margin scale, temperature, Sinkhorn coefficient, learned metric parameter, or external-prior weight. The block-normalized averaging, strict correction, null cost, and multilabel scaling are fixed by (11), (13), (21), and (26), respectively.

## 4 Experiments

## 4.1 Experimental Setting

4.1.1 Dataset. We conducted experiments on the 3DSSG dataset [33], a semantically enriched 3D scene graph dataset built upon 3RScan [32]. 3DSSG provides object instances, semantic object categories, and directed object-object relation annotations for real-world indoor RGB-D reconstructions. We used two label settings that are commonly adopted in recent 3D scene graph prediction studies.

For the multi-label setting, following recent works [10, 34], we used the original 3DSSG label space with 160 object categories and 26 predicate types, together with the standard train/validation split over 1,553 indoor scenes. In this setting, an ordered object pair may be associated with multiple relation labels.

For the single-label setting, we used the 3DSSG-O27R16 label space and train/validation split introduced by SGGPoint [37]. This setting is based on the RIO27 object annotation and maps the original object vocabulary into 27 object classes and the relation vocabulary into 16 relation classes. The detailed single-label conversion and scoring protocol are described in Sec. 4.1.2, since they afect how predictions are evaluated rather than the source dataset itself.

4.1.2 Metrics and Task Description. Because the multi-label and single-label settings use diferent label spaces and scoring rules, we report their results separately.

Multi-label Setting. We follow the standard 3DSSG evaluation protocol [33, 38]. We evaluate object, predicate, and triplet prediction using top-� Recall (R@�) and top-� mean Recall (mR@�). For triplet prediction, a predicted triplet is counted as correct only when its subject category, predicate label, and object category all match a ground-truth triplet. We report results on two standard scene graph prediction tasks [36]: Predicate Classification (PredCls) and Scene Graph Classification (SGCls). In PredCls, ground-truth object instances and object categories are given, and the model predicts predicates for directed object pairs. In SGCls, ground-truth object instances are given, and the model predicts both object categories and predicates, so the final subject-predicate-object triplets are evaluated.

Single-label Setting. We follow the 3DSSG-O27R16 protocol used in recent single-label methods [22, 37], where comparative predicates are removed, directional proximity predicates are merged into a unified spatial proximity relation, and ambiguous multi-label relation annotations are converted into a single target relation label. We evaluate object prediction and relationship prediction with R@�, and predicate prediction with F1@�. Unlike recall alone, F1@� jointly reflects precision and recall, and therefore penalizes both false positive predicate predictions and missed predicate labels. This is especially relevant in the single-label setting, where each directed object pair is represented by one relation target.

These automatic metrics are based on exact label agreement. They are therefore appropriate for comparison with prior work, but they do not directly measure whether an incorrect label is semantically close to the ground truth or whether the predicted scene graph remains plausible as a whole. We thus complement the quantitative benchmark results with user studies in Sec. 4.4.

4.1.3 Implementation Details. All experiments were conducted on a single NVIDIA RTX 6000 Ada GPU with 48GB memory. We trained all models for 100 epochs using the AdamW optimizer [15, 20] with a global batch size of 8 scenes and an initial learning rate of 1×10<sup>−4</sup>. The learning rate was decayed with a cosine schedule [19]. For each relation setting, we followed the original implementation of the corresponding backbone in terms of architecture, input preprocessing, training configuration and inference pipeline, while replacing only the supervision objective with the proposed SWORD objective. The object and relation distance matrices were computed once from the training split and kept fixed during training. During training, the precomputed matrices were used only through distance lookup in the corresponding SWORD training objective, either as Wasserstein ground costs or as semantic cost margins.

Table 1: Object, predicate, and triplet recall on 3DSSG in the multi-label setting.
<table><tr><td rowspan="2">Model</td><td colspan="4">Object</td><td colspan="4">Predicate</td><td colspan="4">Triplet</td></tr><tr><td>R@1</td><td>R@5</td><td>mR@1</td><td>mR@5</td><td>R@1</td><td>R@3</td><td>mR@1</td><td>mR@3</td><td>R@50</td><td>R@100</td><td>mR@50</td><td>mR@100</td></tr><tr><td>SGPN [33]</td><td>49.46</td><td>73.99</td><td>10.86</td><td>27.54</td><td>86.92</td><td>94.76</td><td>32.01</td><td>55.22</td><td>85.38</td><td>88.59</td><td>41.52</td><td>51.92</td></tr><tr><td>SGFN [35]</td><td>53.36</td><td>76.88</td><td>21.24</td><td>46.68</td><td>89.00</td><td>97.71</td><td>41.89</td><td>70.82</td><td>88.59</td><td>91.14</td><td>58.37</td><td>67.61</td></tr><tr><td>VL-SAT [34]</td><td>55.93</td><td>78.06</td><td>21.41</td><td>46.14</td><td>89.81</td><td>98.46</td><td>54.03</td><td>77.67</td><td>89.35</td><td>92.20</td><td>65.09</td><td>73.59</td></tr><tr><td>OCRL [10]</td><td>59.53</td><td>81.20</td><td>22.55</td><td>48.10</td><td>91.27</td><td>98.48</td><td>56.32</td><td>76.26</td><td>91.40</td><td>93.80</td><td>65.31</td><td>74.54</td></tr><tr><td>SWORD</td><td>59.86</td><td>82.58</td><td>24.08</td><td>50.84</td><td>91.38</td><td>99.02</td><td>58.74</td><td>78.92</td><td>92.36</td><td>94.82</td><td>68.41</td><td>78.06</td></tr></table>

Table 2: PredCls and SGCls results on 3DSSG in the multi-label setting.
<table><tr><td rowspan="2">Model</td><td colspan="6">PredCls</td><td colspan="6">SGCls</td></tr><tr><td>R@20</td><td>R@50</td><td>R@100</td><td>mR@20</td><td>mR@50</td><td>mR@100</td><td>R@20</td><td>R@50</td><td>R@100</td><td>mR@20</td><td>mR@50</td><td>mR@100</td></tr><tr><td>SGPN [33]</td><td>51.9</td><td>58.0</td><td>58.5</td><td>32.1</td><td>38.4</td><td>38.9</td><td>27.0</td><td>28.8</td><td>29.0</td><td>19.7</td><td>22.6</td><td>23.1</td></tr><tr><td>SGFN [35]</td><td>65.9</td><td>78.8</td><td>79.6</td><td>46.1</td><td>54.8</td><td>55.1</td><td>29.5</td><td>31.2</td><td>31.2</td><td>20.5</td><td>23.1</td><td>23.1</td></tr><tr><td>VL-SAT [34]</td><td>67.8</td><td>79.9</td><td>80.8</td><td>57.8</td><td>64.2</td><td>64.3</td><td>32.0</td><td>33.5</td><td>33.7</td><td>31.0</td><td>32.6</td><td>32.7</td></tr><tr><td>OCRL [10]</td><td>70.2</td><td>82.0</td><td>82.6</td><td>58.8</td><td>66.1</td><td>66.4</td><td>36.1</td><td>37.7</td><td>37.8</td><td>35.3</td><td>37.7</td><td>37.9</td></tr><tr><td>SWORD</td><td>71.6</td><td>83.5</td><td>84.1</td><td>61.4</td><td>69.0</td><td>69.4</td><td>37.2</td><td>39.0</td><td>39.2</td><td>37.1</td><td>39.7</td><td>40.1</td></tr></table>

## 4.2 Comparison with Related Methods

We evaluated SWORD through comparisons with recent 3D scene graph prediction methods. Since SWORD supports both multi-label and single-label relation prediction, we reported the results separately for the two settings.

4.2.1 Multi-Label Seting. For the multi-label setting, we compared SWORD with SGPN [33], SGFN [35], VL-SAT [34], and OCRL [10]. To isolate the efect of the proposed supervision, we used OCRL as the backbone and replaced only the loss-side training objective with the multi-label SWORD objective in Eq. (33). The input modality, network architecture, graph construction, and evaluation protocol are kept unchanged. Thus, this comparison directly reflects the benefit of introducing scene-graph-grounded object and relation costs into training.

Table 1 reports object, predicate, and triplet prediction results. Among the compared methods under the same multi-label evaluation protocol, SWORD achieves the strongest performance across all reported metrics. This is notable because the evaluation still uses exact ground-truth labels: SWORD does not relax the test criterion, but improves strict label ranking by changing the geometry of the training signal. Compared with OCRL, the most meaningful trend appears in mean-recall and triplet-level measures. Mean recall is less dominated by frequent classes, so the consistent improvement suggests that SWORD provides more informative supervision for rare, ambiguous, or visually confusable object-relation patterns. This behavior is consistent with the design of SWORD: semantically nearby substitutions receive smaller graph-grounded costs than distant substitutions, while the underlying SWORD distance still preserves the ground-truth graph as the unique zero-distance solution. As a result, the model can organize its prediction distribution around plausible semantic neighborhoods without giving up exact-label optimality. The triplet results further show that the improvement is not limited to isolated object or predicate classifi cation. Since a triplet is correct only when the subject, predicate, and object are jointly aligned, the strong triplet performance indicates that SWORD improves the compatibility of node and edge predictions at the graph level. This supports the central motivation of SWORD: object and relation labels should not be supervised as independent local decisions only, but should be structured by their scene-graph-grounded semantic proximity.

Table 2 shows the PredCls and SGCls results. In PredCls, object categories are given, so the task emphasizes relation reasoning. SWORD consistently outperforms OCRL, indicating that the proposed relation distance improves predicate prediction itself rather than only benefiting from better object recognition. In SGCls, both object classification and relation prediction are involved, and SWORD again maintains the best overall performance. This trend suggests that the object and relation ground metrics work coherently: the object-side margin improves node predictions, while the multi-label Wasserstein relation loss provides structured supervision over predicate labels and the null relation.

Figure 2 provides qualitative evidence for this behavior. In the bathroom scene, OCRL introduces objects such as microwave and toilet, and produces relations such as part of and behind that are less consistent with the visible scene structure. SWORD still makes some exact-label mismatches, such as predicting bath cabinet or front, but these predictions remain closer to the bathroom context and preserve most of the reference object-relation layout. In the desk scene, OCRL replaces scene-consistent objects with tv and monitor and assigns an implausible hanging on relation to the table. In contrast, SWORD preserves the window-lamp-table-chair arrangement and predicts a physically compatible support relation between the lamp and the table. These examples illustrate that SWORD tends to avoid semantically distant substitutions and yields more coherent graphs when exact labels are uncertain.

Overall, the multi-label results demonstrate that SWORD improves both strict quantitative metrics and qualitative graph coherence. By deriving object and relation distances from training scene graphs and using them as supervision geometry, SWORD encourages predictions that are not only more accurate under exactlabel evaluation but also more consistent with the semantic and structural organization of 3D scenes.

![](images/5c9f354f718deaab3e2d5bbb241460d72cbba583d98667a8dfffc6d2b8473c2f.jpg)  
Figure 2: Qualitative comparison in the multi-label setting. Red nodes and edges denote deviations from the reference graph.

Table 3: Quantitative comparison under the standard singlelabel protocol [22, 37].
<table><tr><td rowspan="2">Model</td><td colspan="2">Object</td><td colspan="2">Predicate</td><td colspan="2">Relationship</td></tr><tr><td>R@5</td><td>R@10</td><td>F1@3</td><td>F1@5</td><td>R@50</td><td>R@100</td></tr><tr><td>SGGPoint [37]</td><td>90.70</td><td>97.58</td><td>78.88</td><td>90.86</td><td>39.91</td><td>48.68</td></tr><tr><td>SGFormer [22]</td><td>96.41</td><td>98.48</td><td>85.12</td><td>91.58</td><td>56.25</td><td>60.67</td></tr><tr><td>SWORD (Multi)</td><td>96.48</td><td>98.55</td><td>85.30</td><td>91.74</td><td>56.58</td><td>60.94</td></tr><tr><td>SWORD (Single)</td><td>97.20</td><td>98.82</td><td>86.95</td><td>92.72</td><td>58.25</td><td>62.55</td></tr></table>

4.2.2 Single-Label Seting. To validate the proposed method in the single-label setting, we compared it with recent single-label methods: SGGPoint [37] and SGFormer [22]. We adopted SGFormer as the backbone and replaced its original training objective with the proposed single-label SWORD-Margin loss in Eq. (32). We also evaluated the multi-label SWORD on the same label space under the single-label protocol by training it with the sigmoid-based multilabel SWORD objective and selecting the predicate with the highest sigmoid output for each directed object pair.

Table 3 shows that the multi-label variant already improves over SGFormer across all reported criteria, indicating that the datasetinternal object-relation geometry ofSWORD is beneficial even without a single-label training objective. The single-label variant further achieves the best performance in every column, with a clearer advantage on predicate and relationship evaluation. Since the graph encoder is kept fixed, this gain is attributable to the supervision geometry introduced by SWORD-Margin rather than to an architectural change. This behavior is consistent with the structure of the objective. In the single-label setting, relation candidates are mutually exclusive, and the correct prediction must be selected jointly against other predicates and the no-relation state. SWORD-Margin in Eq. (32) matches this assumption by applying a cost-augmented softmax over ${ \mathcal { R } } _ { 0 } ,$ where incorrect labels are separated according to the scene-graph-grounded relation distance. Thus, semantically distant alternatives are discouraged more strongly, while nearby alternatives remain ordered by the learned graph geometry. In con trast, the multi-label variant predicts each predicate independently with sigmoid outputs. This formulation is natural when multiple relations may hold for the same object pair, but under one-label decoding the scores are not calibrated as mutually exclusive choices. As a result, generic or frequent predicates can remain competitive, and the null state is represented through induced probability mass rather than an explicit competing logit.

Figure 3 provides qualitative evidence for this interpretation. SGFormer often recovers part of the scene structure, but its errors tend to weaken the specificity of object-relation configurations. In the first scene, it replaces blanket with towel and maps bed-centered physical relations to the broader predicate close by. In the second scene, an object error from shelf to fridge leads to an implausible relation such as clothes part offridge. SWORD preserves the main object identities and relation structure more consistently. When it difers from the reference, as in pillow leaning against bed or cabinet hanging on wall, the predicted relation remains compatible with the local geometry and is closer to the reference than a structurally inconsistent object-relation substitution.

Overall, these results consistently suggest that SWORD-Margin turns the scene-graph-grounded object and relation metrics into an efective training signal for single-label 3D scene graph prediction. It improves exact-label evaluation while preserving the intended behavior of SWORD: the reference graph remains the optimal solution, but graph-grounded near misses are distinguished from semantically and structurally inconsistent predictions.

## 4.3 Ablation Study

We analyzed the two key components of SWORD: the Wassersteinbased supervision objective in Sec. 3.3 and the scene-graph-grounded ground metrics in Sec. 3.2. All variants used the OCRL backbone and follow the same multi-label setting, dataset, tasks, and evaluation protocol as in Sec. 4.1. We only change either the training objective or the label distances used in the Wasserstein terms.

![](images/89ad57892155c1551d9af4049b4d0dfa1604d6450f31e7212e68d33c4397b060.jpg)  
Figure 3: Qualitative comparison in the single-label setting. Red nodes and edges denote deviations from the reference graph.

For the ground-metric variants, we keep the proposed multilabel SWORD objective in Eq. (33) fixed and vary only the ground distance. Uniform Dist. uses a constant distance for all distinct labels, and Text-Emb. Dist. computes label distances from CLIP text embeddings [25] of object and relation names. Base (w/o Context) uses the base metrics �<sup>base</sup> and �<sup>base</sup> defined in Eqs. (14) and (16). For the objective variants, we keep the context-aware metrics �<sub>�</sub> and �<sub>�</sub> in Eqs. (18) and (20) fixed. SWORD + CE/BCE adds the original OCRL CE/BCE losses to Eq. (33) with unit coeficients and without tuning an additional mixing weight, whereas SWORD (Ours) uses the proposed SWORD objective without the CE/BCE terms.

Table 4 shows a consistent trend across object, predicate, triplet, PredCls, and SGCls evaluations. Uniform Dist. is the weakest among the ground-metric variants, indicating that merely replacing CE/BCE with a transport-style objective is not suficient when the ground cost does not encode meaningful label geometry. Text-Emb. Dist. improves this behavior, suggesting that non-uniform semantic distances are useful. However, it remains below the graph-grounded metrics because language-level similarity between class names does not explicitly capture how object categories participate as relation endpoints, which predicates they co-occur with, or what structural roles they play in 3D scene graphs.

Base (w/o Context) further improves the results by constructing distances from dataset-internal object-relation statistics rather than external language priors. The full context-aware metric, i.e., SWORD (Ours), consistently performs best, showing that the additional context terms refine the label geometry beyond simple graphtoken profiles. In particular, object labels are compared through their neighbor objects, incoming and outgoing relations, relationneighbor contexts, and typed-degree profiles, while relation labels are compared through endpoint configurations, reverse relations, and local relation co-occurrence. This provides supervision that penalizes graph-semantically distant errors more strongly while allowing semantically plausible alternatives to remain closer in the induced metric space. The benefit is also reflected in mean-recall metrics, suggesting that the learned supervision is helpful beyond frequent dominant labels.

The objective comparison shows that SWORD + CE/BCE underperforms the proposed SWORD objective alone. CE/BCE imposes independent exact-label supervision on nodes and edges, whereas SWORD organizes prediction distributions according to the scenegraph-grounded distances. Adding CE/BCE can therefore dilute the distance-aware signal induced by the SWORD metric, especially when predictions should be shaped by semantic proximity rather than only by one-hot or multi-hot targets. Overall, the ablation confirms that SWORD requires both components: a metric-induced objective aligned with Eq. (33) and context-aware ground distances constructed from the internal structure of training scene graphs.

## 4.4 User Study

The quantitative comparisons in Sec. 4.2 show that SWORD improves standard exact-label metrics, even though it changes the supervision geometry rather than simply optimizing uniform CE/BCEstyle local classification penalties. However, exact agreement with reference annotations has an inherent limitation: all incorrect labels are counted as errors, regardless of whether they are graphgrounded near misses or semantically distant substitutions. Therefore, the standard metrics can underrepresent improvements in plausibility, descriptiveness, and structural coherence. To evaluate these aspects, we conducted a blind A/B user study comparing OCRL [10] and SWORD.

We used two complementary settings. In the Graph-Graph setting, participants were given the reference graph and two predicted graphs, and selected the prediction that better satisfied Semantic Closeness and Structure Preservation. In the Scene-Graph setting, participants were given the reference 3D scene and two predicted graphs, and selected the prediction that better satisfied Scene Descriptiveness and Relation Plausibility. A total of 28 participants with normal or corrected-to-normal vision evaluated 20 validation sam ples. Each participant evaluated both settings, and each question was answered as a binary preference without ties; therefore, each criterion was computed over 28 × 20 = 560 pairwise judgments. The order of the two methods was randomized for each sample. All graphs were visualized with the same layout style, node and edge appearance, font, and label format to avoid method-unrelated visual bias. Table 5 reports the preference ratios for each criterion.

Table 4: Ablation study of ground metrics and objectives in SWORD.
<table><tr><td></td><td colspan="2">Object</td><td colspan="2">Predicate</td><td colspan="2">Triplet</td><td colspan="2">PredCls</td><td colspan="2">SGCls</td></tr><tr><td>Ground Metric Variants</td><td>R@5</td><td>mR@5</td><td>R@3</td><td>mR@3</td><td>R@50</td><td>mR@50</td><td>R@50</td><td>mR@50</td><td>R@50</td><td>mR@50</td></tr><tr><td>Uniform Dist.</td><td>80.74</td><td>47.92</td><td>98.25</td><td>76.08</td><td>91.05</td><td>65.02</td><td>81.6</td><td>65.8</td><td>37.3</td><td>37.5</td></tr><tr><td>Text-Emb. Dist.</td><td>81.62</td><td>49.24</td><td>98.61</td><td>77.15</td><td>91.67</td><td>66.48</td><td>82.5</td><td>67.2</td><td>38.0</td><td>38.3</td></tr><tr><td>Base (w/o Context)</td><td>81.90</td><td>49.80</td><td>98.78</td><td>77.84</td><td>91.86</td><td>67.38</td><td>82.8</td><td>68.0</td><td>38.4</td><td>39.0</td></tr><tr><td colspan="9">Training Objective Variants</td><td></td></tr><tr><td>SWORD + CE/BCE Loss</td><td>81.55</td><td>48.72</td><td>98.56</td><td>76.74</td><td>91.55</td><td>65.78</td><td>82.2</td><td>66.5</td><td>37.8</td><td>38.0</td></tr><tr><td>SWORD (Ours)</td><td>82.58</td><td>50.84</td><td>99.02</td><td>78.92</td><td>92.36</td><td>68.41</td><td>83.5</td><td>69.0</td><td>39.0</td><td>39.7</td></tr></table>

Graph-Graph Comparison. SWORD is clearly preferred over OCRL for both graph-based criteria. This result complements the exactlabel evaluation in Sec. 4.2: while the standard metrics show consistent gains in strict object, predicate, and triplet prediction, the human comparison reveals a larger advantage in the semantic and structural quality of the predicted graphs. This gap is expected because exact-label metrics do not distinguish whether an error is a scene-consistent near miss or a semantically disruptive substitu tion. In contrast, SWORD explicitly orders object and relation labels by scene-graph-grounded distances, keeping predictions closer to the reference under exact-label uncertainty. The preference for Structure Preservation further suggests that SWORD better maintains coherent subject-predicate-object organization, rather than improving only isolated node or edge labels.

Scene-Graph Comparison. SWORD is consistently preferred when participants judge the predicted graphs directly against the reference 3D scene. The preference for Scene Descriptiveness indicates that SWORD better captures the object roles and overall scene organization perceived by humans. The preference for Relation Plausibility further suggests that its predicted relations are more compatible with the observed arrangement of objects. This is consistent with the design of SWORD: although its metric construction does not use external semantic priors or raw geometric features, it encodes endpoint contexts, local neighborhoods, typed-degree patterns, and relation co-occurrences from training scene graphs. These graph-grounded profiles guide the model away from semantically distant or structurally implausible alternatives.

Table 5: User study results for Graph-Graph and Scene-Graph comparisons. Participants compared the baseline (i.e., OCRL [10]) and ours under two questions for each setting.
<table><tr><td>Setting</td><td>Question</td><td>| Baseline (%) SWORD (%)</td><td></td></tr><tr><td rowspan="2">Graph-Graph</td><td>Semantic Closeness</td><td>26.1</td><td>73.9</td></tr><tr><td>Structure Preservation</td><td>24.1</td><td>75.9</td></tr><tr><td rowspan="2">Scene-Graph</td><td>Scene Descriptiveness</td><td>25.7</td><td>74.3</td></tr><tr><td>Relation Plausibility</td><td>21.6</td><td>78.4</td></tr></table>

Overall, the user study highlights a key benefit that is only partially visible in conventional metrics. SWORD improves strict exactlabel evaluation, but its larger advantage appears in the quality of the predicted graph as a semantic and structural description of the scene. These results support the central motivation of SWORD: replacing uniform penalties for all wrong labels with scene-graphgrounded distances mitigates annotation ambiguity while preserving exact ground-truth optimality.

## 5 Conclusion

We introduced SWORD, a scene-grounded Wasserstein object-relation distance for 3D scene graph prediction. By exploiting the fixed node and edge-slot correspondence in supervised protocols, SWORD defines an aligned graph distance, constructs object and relation ground metrics from training scene graph statistics, and provides distance-aware supervision beyond CE/BCE-style exact-label losses. This preserves the ground-truth graph as the unique zero-distance solution while distinguishing graph-grounded near misses from semantically distant errors. Experiments on 3DSSG show that SWORD improves object, predicate, relationship, and triplet prediction un der both multi-label and single-label protocols. User studies further indicate that the predicted graphs are more semantically close, structurally preserved, scene-descriptive, and relation-plausible. These results show that SWORD improves not only exact-label benchmark accuracy but also perceived graph quality. Although SWORD demonstrates strong performance, like most 3D scene graph prediction scenarios, it cannot be applied of-the-shelf to settings where node correspondence, edge slots, or label vocabularies are not fixed. Future work will extend SWORD to partially aligned, open-vocabulary, and temporal scene graphs, while exploring geometric structure and perceptual 3D similarity for richer ground metrics [17, 18].

## References

[1] Iro Armeni, Zhi-Yang He, JunYoung Gwak, Amir R Zamir, Martin Fischer, Jitendra Malik, and Silvio Savarese. 2019. 3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 5663–5672.

[2] Horst Bunke. 1997. On a relation between graph edit distance and maximum common subgraph. Pattern Recognition Letters 18, 8 (1997), 689–694.

[3] Lianggangxu Chen, Xuejiao Wang, Jiale Lu, Shaohui Lin, Changbo Wang, and Gaoqi He. 2024. CLIP-driven open-vocabulary 3D scene graph generation via cross-modality contrastive learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 27863–27873.

[4] Tianshui Chen, Weihao Yu, Riquan Chen, and Liang Lin. 2019. Knowledge embedded routing network for scene graph generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 6163–6171.

[5] Donatello Conte, Pasquale Foggia, Carlo Sansone, and Mario Vento. 2004. Thirty years of graph matching in pattern recognition. International Journal ofPattern Recognition and Artificial Intelligence 18, 3 (2004), 265–298.

[6] Mingtao Feng, Haoran Hou, Liang Zhang, Zijie Wu, Yulan Guo, and Ajmal Mian. 2023. 3D spatial multimodal knowledge accumulation for scene graph prediction in point cloud. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9182–9191.

[7] Xinbo Gao, Bing Xiao, Dacheng Tao, and Xuelong Li. 2010. A survey of graph edit distance. Pattern Analysis and Applications 13, 1 (2010), 113–129.

[8] Alison L Gibbs and Francis Edward Su. 2002. On choosing and bounding proba bility metrics. International Statistical Review 70, 3 (2002), 419–435.

[9] Kevin Gimpel and Noah A Smith. 2010. Softmax-Margin CRFs: Training Log-Linear Models with Cost Functions. In Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics. Association for Computational Linguistics, Los Angeles, California, 733–736.

[10] KunHo Heo, GiHyeon Kim, SuYeon Kim, and MyeongAh Cho. 2025. Object-Centric Representation Learning for Enhanced 3D Semantic Scene Graph Predic tion. Advances in Neural Information Processing Systems (2025).

[11] Yucheng Huang, Luping Ji, Xiangwei Jiang, Wen Li, and Mao Ye. 2026. ToLL: Topological Layout Learning with Asymmetric Cross-View Structural Distillation for 3D Scene Graph Pretraining. arXiv preprint arXiv:2603.28178 (2026).

[12] U-Chae Jun, Jaeeun Ko, and Jiwoo Kang. 2026. UniCross3D: Unified Cross-View and Cross-Domain Difusion for Consistent Single-Image 3D Generation. In Computer Graphics Forum. Wiley Online Library, e70378.

[13] Jiwoo Kang, Seongmin Lee, Mingyu Jang, and Sanghoon Lee. 2021. Gradient Flow Evolution for 3D Fusion from a Single Depth Sensor. IEEE Transactions on Circuits and Systems for Video Technology 32, 4 (2021), 2211–2225.

[14] Jiwoo Kang, Seongmin Lee, and Sanghoon Lee. 2021. Competitive Learning of Facial Fitting and Synthesis using UV Energy. IEEE Transactions on Systems, Man, and Cybernetics: Systems 52, 5 (2021), 2858–2873.

[15] Diederik P Kingma and Jimmy Ba. 2015. Adam: A method for stochastic optimiza tion. In Proceedings ofthe International Conference on Learning Representations.

[16] Sebastian Koch, Narunas Vaskevicius, Mirco Colosi, Pedro Hermosilla, and Timo Ropinski. 2024. Open3DSG: Open-vocabulary 3D scene graphs from point clouds with queryable objects and open-set relationships. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14183–14193.

[17] Seongmin Lee, Jiwoo Kang, and Sanghoon Lee. 2025. 3D Facial Shape Similarity with Deep Perceptual Representations. ACM Transactions on Multimedia Computing, Communications and Applications 21, 6 (2025), 1–27.

[18] Seongmin Lee, Jiwoo Kang, Sanghoon Lee, Weisi Lin, and Alan Conrad Bovik. 2024. 3D-PSSIM: Projective Structural Similarity for 3D Mesh Quality Assessment Robust to Topological Irregularities. IEEE transactions on pattern analysis and machine intelligence 46, 12 (2024), 9595–9611.

[19] Ilya Loshchilov and Frank Hutter. 2016. SGDR: Stochastic gradient descent with warm restarts. Proceedings of the International Conference on Learning Representations.

[20] Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In Proceedings ofthe International Conference on Learning Representations.

[21] Cewu Lu, Ranjay Krishna, Michael Bernstein, and Li Fei-Fei. 2016. Visual relationship detection with language priors. In Proceedings of the European Conference on Computer Vision. Springer, 852–869.

[22] Changsheng Lv, Mengshi Qi, Xia Li, Zhengyuan Yang, and Huadong Ma. 2024. SGFormer: Semantic Graph Transformer for Point Cloud-Based 3D Scene Graph Generation. In Proceedings ofthe AAAIConference on Artificial Intelligence, Vol. 38. 4035–4043.

[23] Yanni Ma, Hao Liu, Yulan Guo, Theo Gevers, and Martin R Oswald. 2026. Edge-Centric Relational Reasoning for 3D Scene Graph Prediction. In Proceedings of the AAAI Conference on Artificial Intelligence. 7847–7855.

[24] Yanni Ma, Hao Liu, Yun Pei, and Yulan Guo. 2024. Heterogeneous graph learning for scene graph prediction in 3D point clouds. In Proceedings of the European Conference on Computer Vision. Springer, 274–291.

[25] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In Proceedings ofthe International Conference on Machine Learning. PmLR, 8748– 8763.

[26] Kaspar Riesen and Horst Bunke. 2009. Approximate graph edit distance computation by means of bipartite graph matching. Image and Vision Computing 27, 7 (2009), 950–959.

[27] Kaspar Riesen, Xiaoyi Jiang, and Horst Bunke. 2010. Exact and Inexact Graph Matching: Methodology and Applications. In Managing and Mining Graph Data, Charu C. Aggarwal and Haixun Wang (Eds.). Advances in Database Systems, Vol. 40. Springer, 217–247.

[28] Alberto Sanfeliu and King-Sun Fu. 1983. A distance measure between attributed relational graphs for pattern recognition. IEEE Transactions on Systems, Man, and Cybernetics 13, 3 (1983), 353–362.

[29] Ioannis Tsochantaridis, Thorsten Joachims, Thomas Hofmann, Yasemin Altun, and Yoram Singer. 2005. Large Margin Methods for Structured and Interdependent Output Variables. Journal ofMachine Learning Research 6, 9 (2005), 1453–1484.

[30] Peter D Turney and Patrick Pantel. 2010. From frequency to meaning: Vector space models of semantics. Journal ofArtificial Intelligence Research 37 (2010), 141–188.

[31] Cédric Villani. 2009. Optimal Transport: Old and New. Grundlehren der mathematischen Wissenschaften, Vol. 338. Springer, Berlin.

[32] Johanna Wald, Armen Avetisyan, Nassir Navab, Federico Tombari, and Matthias Nießner. 2019. RIO: 3D object instance re-localization in changing indoor environments. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision. 7658–7667.

[33] Johanna Wald, Helisa Dhamo, Nassir Navab, and Federico Tombari. 2020. Learn ing 3D semantic scene graphs from 3D indoor reconstructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 3961–3970.

[34] Ziqin Wang, Bowen Cheng, Lichen Zhao, Dong Xu, Yang Tang, and Lu Sheng. 2023. VL-SAT: Visual-linguistic semantics assisted training for 3D semantic scene graph prediction in point cloud. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 21560–21569.

[35] Shun-Cheng Wu, Johanna Wald, Keisuke Tateno, Nassir Navab, and Federico Tombari. 2021. SceneGraphFusion: Incremental 3D scene graph prediction from RGB-D sequences. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 7515–7525.

[36] Danfei Xu, Yuke Zhu, Christopher B Choy, and Li Fei-Fei. 2017. Scene graph generation by iterative message passing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 5410–5419.

[37] Chaoyi Zhang, Jianhui Yu, Yang Song, and Weidong Cai. 2021. Exploiting Edge-Oriented Reasoning for 3D Point-Based Scene Graph Analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 9705–9715.

[38] Shoulong Zhang, Shuai Li, Aimin Hao, and Hong Qin. 2021. Knowledge-inspired 3D scene graph prediction in point cloud. Advances in Neural Information Processing Systems 34 (2021), 18620–18632.