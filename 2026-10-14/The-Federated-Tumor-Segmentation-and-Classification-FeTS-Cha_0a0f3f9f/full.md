# The Federated Tumor Segmentation and Classification (FeTS++) Challenge

Wenyi Tang<sup>1,2,3,</sup> , Carla Pitarch-Abaigar<sup>1,2,3,</sup> , Siddhesh Thakur<sup>1,2,3,</sup> , Verena Chung<sup>4,</sup> , Konstantinos Koukoutegos<sup>1,2,3,</sup> , Alexandros Karargyris<sup>5,</sup> , Holger R. Roth<sup>6,</sup> , and Spyridon Bakas<sup>1,2,3,5,7,8,∗,</sup>

1 Research Center for Federated Learning in Medicine, Indiana University School of Medicine, Indianapolis, IN, USA.

<sup>2</sup> Division of Computational Pathology, Department of Pathology and Laboratory Medicine, Indiana University School of Medicine, Indianapolis, IN, USA.

3 Indiana University Melvin and Bren Simon Comprehensive Cancer Center, Indianapolis, IN, USA.

4 Sage Bionetworks, Seattle, WA, USA.

<sup>5</sup> MLCommons, Dover, DE, USA.

<sup>6</sup> NVIDIA, Santa Clara, CA, USA.

Departments of Radiology and Imaging Sciences; Neurological Surgery; Biostatistics and Health Data Science, Indiana University School of Medicine, Indianapolis, IN, USA.

<sup>8</sup> Department of Computer Science, Luddy School of Informatics, Computing, and Engineering, Indiana University, Indianapolis, IN, USA Corresponding author: {spbakas@iu.edu}

Abstract. International challenges have become the standard for developing and benchmarking biomedical image analysis methods using centralized multi-institutional data. Federated learning (FL) describes a paradigm-shift on multi-institutional collaborations, where models are trained across institutions while their data are always retained locally. The Federated Tumor Segmentation (FeTS) challenge, first held in 2021, was the first challenge dedicated to FL in medical imaging and editions in 2022 and 2024 pushed the frontiers of real-world method validation. FeTS++ 2027 returns after a two-year gap with a substantially larger and more diverse cohort, and with the two ”pluses” that name the challenge: an oficial classification task alongside segmentation, and a move of the infrastructure to NVIDIA FLARE (NVFlare). Built on the clinically acquired, multi-institutional multi-parametric MRI data of the BraTS 2025 Lighthouse challenge, FeTS++ spans gliomas and meningiomas across patient cohorts from institutions in North America, Europe, Asia, and Africa, and focuses on ranking submitted methodological approaches according to their model utility and an eficiency-aware score. Task 1 asks participants to find the best strategy for aggregating knowledge from per-institution segmentation models, while Task 2 asks them to distinguish glioma from meningioma under the same federated protocol, in which the entity label is heavily confounded with the site of origin.

Keywords: federated learning, segmentation, domain generalization, deep learning, convolutional neural network, brain tumor, glioma, glioblastoma, meningioma, BraTS, FeTS

## 1 Introduction

Gliomas and meningiomas are the most common primary brain tumors in adults. Gliomas are the most common malignant tumors of the central nervous system (CNS), whereas meningiomas are the most common intracranial tumors. Glioblastoma, representing the most aggressive form of glioma, accounts for more than half of all primary malignant CNS tumors [1] and is characterized by extreme intrinsic heterogeneity in appearance, shape, and histology [2]. Despite standard of care comprising surgical resection followed by radiotherapy and temozolomide, median overall survival remains around 15 months [3]. This poor prognosis is largely due to its difusely infiltrative growth beyond the radiologically visible margins, which makes tumor recurrence almost inevitable. Meningiomas on the other hand, the most common primary intracranial tumors in adults, represent over 40% of all primary brain and other CNS tumors [1]. Approximately 80% are CNS WHO grade 1 tumors that are typically well controlled with observation, surgical resection, and/or radiotherapy. However, grade 2 and 3 meningiomas are associated with significantly higher morbidity and mortality and often recur despite optimal management [4].

Multiparametric magnetic resonance imaging (mpMRI), typically comprising T1-weighted (T1), post-contrast T1-weighted (T1-Gd), T2-weighted (T2), and T2-FLAIR sequences, is the standard of care for diagnosis, surgical and radiotherapy planning, and response assessment. Accurate delineation of tumor sub-regions, including enhancing tumor, necrotic tumor core, and peritumoral edematous tissue, is essential for patient management, yet manual segmentation remains time-consuming and subject to inter-rater variability.

The Brain Tumor Segmentation (BraTS) challenge has served since 2012 as the community benchmark for automated brain tumor analysis on mpMRI [5,6]. Since 2023, BraTS has evolved into a cluster of challenges covering tumors beyond glioma, including meningioma, brain metastases, and pediatric brain tumors [4,7], as well as beyond segmentation workloads changing the definition of BraTS to ‘Brain TumorS’. Nevertheless, BraTS relies on centralized data collection. Data sharing across institutions is hindered by privacy regulations such as Health Insurance Portability and Accountability (HIPAA) and the European Union General Data Protection Regulation (GDPR), along with data ownership and legal considerations, which limit the size and diversity of the datasets available for developing robust and generalizable algorithms [2].

Federated Learning (FL) ofers a solution to this limitation by enabling multiple institutions to collaboratively train a shared model without exchanging patient data. Only model updates are communicated, and the patient data are retained within their local site [8, 9]. Building on this paradigm, the Federated Tumor Segmentation (FeTS) initiative showed that FL can be used in practice to train a glioblastoma segmentation model across dozens of institutions worldwide [10]. Within this initiative, the MICCAI FeTS challenge was launched in 2021 as the very first FL challenge across fields. The MICCAI FeTS challenge focused on benchmarkin FL aggregation and segmentation algorithms on medical imaging data distributed across multiple international institutions [11, 12].

The Federated Tumor Segmentation and Classification (FeTS++) Challenge 2027 is the latest edition of the FeTS series, and its two pluses mark two extensions: i) a tumor classification workload added alongside segmentation, and ii) a migration of the framework infrastructure to NVIDIA FLARE (NVFlare) [13]. The challenge therefore spans two oficial tasks under one federated learning benchmark: federated brain tumor segmentation (Task 1) and federated brain tumor classification (Task 2). Where earlier editions focused on glioblastoma, FeTS++ 2027 covers two tumor entities, namely gliomas and meningiomas, and draws on a substantially larger and more diverse clinically acquired, multiinstitutional dataset of over 3,500 patients from the BraTS 2025 Lighthouse Challenge. Participants develop weight aggregation strategies and client training behavior, whereas the model architecture is fixed so that gains are attributable to the FL aggregation method. Submitted methods will be evaluated on hidden data in a controlled NVFlare simulator that respects the FL paradigm, and will be ranked with an eficiency-aware score using a statistically-aware ranking for fair Benchmarking of AI Models (namely PermRanker) [14].

In summary, FeTS++ brings together to the scientific community:

– A federated brain-tumor benchmark of over 3,500 patients spanning gliomas, meningiomas, and multiple patient populations, including a Sub-Saharan Africa cohort.

– Two oficial tasks under one FL protocol, with locked model architectures so that gains are attributable to the FL aggregation method.

A reproducible NVFlare-/MONAI-based simulator in which public and hidden evaluation share one code path and one frozen evaluator.

An eficiency-aware ranking protocol that combines accuracy with communication cost and pairwise statistical testing.

## 2 Materials & Methods

## 2.1 Data

2.1.1 Data Sources This challenge builds on the BraTS 2025 Lighthouse data [7], publicly released and augmented with a de-identified institution identifier for each training case. It combines three sources: glioma cases carried over from the FeTS 2024 partition, meningioma cases from the meningioma challenge cohort, and a Sub-Saharan Africa cohort [4, 15, 16]. Each case contains clinically acquired multi-parametric structural MRI at a single timepoint, with four co-registered sequences: native T1, T1-Gd, T2, and T2-FLAIR, together with expert reference standard sub-region labels for the training cases.

For this FeTS iteration, challenge datasets are described in the Croissant standard format [17]. Croissant is an open metadata format for machine learning datasets maintained by the MLCommons organization that allows datasets to be better described via schema.org [18]. Croissant hence allows them to be indexed, versioned, and thus findable and reproducible, respectively. The Croissant standard is widely adopted by the machine learning community, and most data repositories and machine learning frameworks support it. In this way discoverability of FeTS++ datasets can be further magnified reaching out to a larger community.

2.1.2 Data Pre-processing The pre-processing pipeline applied to every case follows the one used across the BraTS and FeTS challenges. Raw scans are converted from DICOM to NIfTI [19] and re-oriented to LPS/RAI orientation. All sequences (T1, T1-Gd, T2, T2-FLAIR) are then registered to the SRI-24 atlas [20] and resampled to 1 mm<sup>3</sup>. Registration proceeds as follows: N4 biasfield correction is applied temporarily to estimate the transformation matrices and is then discarded, because it otherwise obliterates the tumor signal [21]. T1-Gd is rigidly registered to the SRI-24 atlas, and then T1, T2, and T2-FLAIR are rigidly registered to T1-Gd. The resulting deformation matrices are then joined and applied to the LPS-oriented scans. After registration, brain extraction removes non-brain tissue such as the skull, neck fat, and eyeballs, using a deeplearning method explicitly developed and evaluated on scans with difuse glioma and meningioma [22]. All scans are then manually checked to confirm that the whole brain is retained and all non-brain tissue is removed.

2.1.3 Annotation Protocol All cohorts were annotated following the BraTS annotator-approver model. A pre-defined clinically approved annotation protocol was provided to all clinical annotators, describing the radiologic appearance each tumor sub-region should delineate on the given MRI sequences. Glioma cases were annotated by 60 clinical neuroradiologists volunteering through the American Society of Neuroradiology (ASNR) and the American Society of Functional Neuroradiology (ASFNR). Each case was assigned to an annotator-approver pair, with the two approvers being board-certified neuroradiologists with more than 13 years of experience. Annotators could use their tool of preference and follow either a complete manual approach or a hybrid approach in which an automated method produced initial annotations that were then refined manually. Each annotation was reviewed by the approver against the corresponding MRI scans and returned to the annotator for refinement until it reached satisfactory quality for sign-of. Meningioma cases followed the same process, with nnU-Net pre-segmentations refined by ASNR/ASFNR volunteers and approved by a board-certified neuroradiologist [15]. Sub-Saharan African cases were refined by local radiology residents and radiologists before final approval by an expert BraTS neuroradiologist [16]. No aggregation of multiple annotations were performed.

The radiologic appearance of each tumor sub-region in MRI, communicated to the radiologists, are as follows:

1. the whole tumor (farthest extent, including the peritumoral edema) delineates the hyperintense abnormal signal on T2 and T2-FLAIR;

2. the tumor core (TC), including the enhancing, non-enhancing, and necrotic tumor, delineates regions of lower T2 signal and represents the resectable bulk of the tumor;

3. the enhancing tumor (ET) delineates the hyperintense signal on T1-Gd, after excluding the vessels;

4. the necrotic core (NCR) outlines regions that appear dark on both T1 and T1-Gd (necrosis/cysts), and regions dark on T1-Gd but bright on T1 (hemorrhage).

Segmentation labels follow the BraTS 2023 convention: label 0 for background, 1 for NCR, 2 for ED, and 3 for ET. The tumor entity label (glioma or meningioma) was derived from the source cohort of each case. Glioma cases were histopathologically confirmed diagnosis [2], except in the Sub-Saharan Africa cohort, where diagnosis was radiological [16]. Meningiomas were diagnosed by imaging or histopathology [15].

The testing datasets are not shared with the challenge participants or the public. The FeTS collaborators collect and process data from their clinical sites following the pre-processing and annotation protocols presented above. From all these data, approximately 80% is kept for federated training (outside the context of this challenge) and the rest is set aside for testing and that will be used to evaluate the performance of submitted algorithms.

## 2.2 Code Framework

2.2.1 Software Stack The challenge is delivered as a simulator-first FL environment built on NVIDIA FLARE (NVFlare)<sup>1</sup>, an open-source, extensible framework developed by NVIDIA [13]. NVFlare coordinates the server and participating clients, distributes the global model, and collects local updates for aggregation. Its modular architecture separates federated orchestration from the underlying model and training code, allowing participants to investigate aggregation strategies and local training behavior within a common execution framework. The NVFlare Client API connects the local training code to this workflow through flare.receive() and flare.send(). The framework also supports deployment across independently operated sites, providing a path for subsequent evaluation beyond the challenge’s controlled simulation environment.

Public local evaluation and oficial hidden evaluation share a single code path and a single frozen evaluator, so that only the data directory difers between them. The data pipeline and the baseline model are provided through MONAI<sup>2</sup> [23], which supplies the image transforms, sliding-window inference, the metric, and a baseline model wrapper.

2.2.2 Locked Editing Policy To keep any improvement attributable to the FL method rather than to the model, the baseline model definition is locked. Participants edit exactly two files: client.py, which exposes local train(), and aggregator.py. Every other file is held fixed and checked against a SHA256 manifest at submission time.

![](images/361a1dadeb8d13c0679b65bc8a5b658a327df6b6ef81b104d10c3530d63cdda4.jpg)  
Fig. 1: One federated round in the FeTS++ 2027 NVFlare simulator. The server broadcasts the global model, each client receives it with flare.receive(), trains it locally with the participant’s local train(), and returns the updated model with flare.send(). The server then aggregates the accepted updates into the next global model, and the cycle repeats while flare.is running(). Blue elements are participant-editable, namely the local training, the aggregation, and the serializable FLModel payload shown at the bottom, which is the only information that crosses the link. Schematic drawn from the challenge starter code, namely the client runner and the participant aggregator. Each simulated site represents a logical client associated with an institutional data partition. The figure illustrates the challenge’s federated execution protocol within the simulator.

Three validity criteria apply: 1) the model architecture must remain unchanged, 2) each simulated client may access only its assigned institutional data partition, and 3) communication must be restricted to the FLModel payloads permitted by the challenge and exchanged through flare.send() and flare.receive(). These payloads may contain model parameters or parameter diferences, metrics, and protocol metadata, but must not carry raw images, reference annotations, or other case-level records. No additional communication channels are permitted.

Figure 1 summarizes one federated round and the two calls that carry it.

2.2.3 Eficiency Federated methods may trade accuracy against communication cost and overall FL training time, so the challenge scores eficiency jointly with task performance. Eficiency is measured as the area under the performanceversus-time (or round) curve, which captures how quickly a method reaches its best accuracy rather than only the final score. Eficiency is normalized relative to the best-performing method and carries 20% of the final ranking, with the full formula given in Section 3.1.3 and Section 3.2.3.

## 3 Challenge Tasks

## 3.1 Task 1: Federated Tumor Segmentation

3.1.1 Task Description Task 1 asks participants to find the best way to aggregate knowledge from segmentation models trained independently at individual institutions, rather than to design a better segmentation network. The task is limited to the methodological parts of FL, i.e., 1) aggregation, 2) client selection, 3) per-round local training, 4) compression, and 5) communication eficiency.

The architecture, data pipeline, splits, initial checkpoint, and federated lifecycle are fixed by the organizers, and the evaluation is hidden while respecting the FL paradigm. Any measured diference between submissions therefore reflects the FL method, not model capacity or data access. Participants work within the two-file editing surface and validity criteria in Section 2.2, and may start from the reference aggregators shipped with the baseline: weighted FedAvg, coordinate-wise median, and clipped mean.

The FeTS++ 2027 cohort is deliberately realistic rather than balanced: persite training counts span two orders of magnitude, and the entity mix is strongly site-dependent. This makes aggregation non-trivial and connects the task to established results on heterogeneous client data [24, 25].

3.1.2 Model Definition and Data Task 1 uses a fixed MONAI SegRes-Net baseline [23, 26] with four input channels, three output channels, an initial filter count of 8, and a dropout probability of 0.2. The baseline checkpoint and data pipeline are provided and locked. The training set comprises the four co-registered mpMRI sequences and their segmentation masks, together with a partition CSV that assigns each case to its original acquisition institution. The cohort is deliberately imbalanced across 26 institutions: per-site case counts range from 4 to 542, and 19 of the 26 sites contribute fewer than 50 cases each. Figure 2 shows how unevenly the training cases are distributed across the 26 sites.

## 3.1.3 Performance Evaluation Submissions will be scored on the following regions:

1. the “enhancing tumor” (ET), label 3;

2. the “tumor core” (TC), labels 1 and 3;

3. the “whole tumor” (WT), labels 1, 2 and 3.

![](images/8c05c41a4a6b1e00cda05af6f3b3cb700f1d3366f0a923e9f8fa71b53294dac6.jpg)  
Fig. 2: Training cases per site in the FeTS++ 2027 cohort. Bars are ordered by decreasing size and the sites are renumbered for presentation, so the indices do not correspond to the identifiers in the released metadata. The dashed line marks 50 cases, below which 19 of the 26 sites fall, and per-site counts span 4 to 542 cases.

The Dice similarity coeficient (DSC) measures the overlap between the predicted mask (PM) and the ground truth (GT) [27]:

$$
{ \cal D } S C = \frac { 2 | G T \cap P M | } { | G T | + | P M | } .\tag{1}
$$

The Normalized Surface Distance (NSD) is used to capture boundary agreement between the prediction and the reference: the boundary voxels of each segmentation that lie within a small distance tolerance of the other boundary are counted as matching, so that minor annotation imprecision is not penalized [28, 29]. NSD lies in [0, 1], where higher values indicate better boundary agreement, so it complements DSC by focusing on the tumor boundary rather than on the enclosed volume.

## 3.2 Task 2: Federated Tumor Classification

3.2.1 Task Description Task 2 asks participants to classify each case as glioma or meningioma under the same FL protocol as Task 1, using the same multi-institutional mpMRI data but without the segmentation masks. Correct entity identification is the first clinical decision that conditions the treatment pathway, and it is exactly the kind of label that is unevenly represented across institutions. Participants use the same two-file editing surface and validity criteria as Task 1 Section 2.2.

The headline dificulty is that the entity label is almost perfectly confounded with the site of origin, so a training algorithm that simply minimizes an aggregate loss aligns with the majority entity and can degrade on minority-entity clients.

This is the classical label-skew problem in FL [25], here with clinical rather than synthetic provenance. Aggregation methods are therefore evaluated not only on average accuracy but on how they behave when the label distribution difers across clients, which makes weighting schemes, client selection, and per-entity calibration all relevant.

3.2.2 Model Definition and Data The classification baseline is fixed and provided. The training inputs are the same four co-registered mpMRI sequences as Task 1, without the segmentation masks, together with a partition CSV and a label CSV that will be released after the segmentation task opens. The entity label is heavily confounded with the site: 18 of the 26 training sites are entity-pure (16 glioma-only, 2 meningioma-only), and the mixed sites are heavily unbalanced. Figure 3 shows the same distribution as the per-site entity composition.

![](images/79119ebb3b7cc74c538796d80b84f822f2070f740dac43380682c49a11a435a2.jpg)  
Fig. 3: Entity composition per site in the FeTS++ 2027 training cohort. Each bar is one site, split into glioma and meningioma cases, in the same renumbered order as Figure 2. 16 sites are glioma-only and two are meningioma-only, as marked by the arrows, and the percentages above the 4 largest mixed sites give their glioma share. Sites are renumbered for presentation, and the indices do not correspond to the identifiers in the released metadata.

3.2.3 Performance Evaluation Submissions will be scored with balanced accuracy and the Matthews correlation coeficient (MCC), two metrics that are insensitive to the class imbalance of the cohort. Balanced accuracy is the mean of the per-class recalls, and MCC measures the chance-corrected agreement between the predictions and the labels. Each metric is ranked across methods, higher being better, with ties broken by rank averaging, and each rank is normalized to a score in [0, 1] by linear rescaling of the mean rank across submissions.

The final score combines balanced accuracy, MCC, and the eficiency term from Section 2.2 as $S _ { i } = 0 . 4 P _ { B A } + 0 . 4 P _ { M C C } + 0 . 2 E _ { i }$ , where $P _ { B A }$ and $P _ { M C C }$ are the normalized ranks of the two metrics and $E _ { i }$ is the normalized convergence score of method i. Accuracy, AUC-ROC, F1, sensitivity, and specificity are reported as diagnostics.

## 4 Discussion and Conclusion

FeTS++ 2027 provides a comprehensive FL benchmark environment and data that spans across two clinical tasks, tumor segmentation and tumor classification, over an international multi-institutional cohort that is deliberately realistic rather than balanced. The shared, locked protocol, under which only the aggregation and client-training logic may change, isolates the contribution of the FL method from the contribution of the model architecture.

Adding classification alongside segmentation stresses a diferent property. Because the entity label is confounded with the site, Task 2 tests whether aggregation methods can serve clients whose case mix difers sharply from the global average. The results will show whether the methods that stabilize segmentation also transfer to classification, or whether the two tasks demand diferent methodological approaches.

This manuscript describes the challenge design and deliberately reports no results. The results and a complete meta-analysis will follow in a joint publication following the completion of the challenge.

## Acknowledgments

Research reported in this publication was partly supported by the National Institutes of Health (NIH) under award numbers NIH/NCI:U01CA242871 and NI-H/NCI:U24CA279629. The content of this publication is solely the responsibility of the authors and does not represent the oficial views of the NIH.

## References

1. M. Price, C. A. P. Ballard, J. R. Benedetti, C. Kruchko, J. S. Barnholtz-Sloan, and Q. T. Ostrom, “Cbtrus statistical report: Primary brain and other central nervous system tumors diagnosed in the united states in 2018–2022,” Neuro-Oncology, vol. 27, pp. iv1–iv66, 10 2025.

2. U. Baid, S. Ghodasara, S. Mohan, M. Bilello, E. Calabrese, E. Colak, K. Farahani, J. Kalpathy-Cramer, F. C. Kitamura, S. Pati, et al., “The rsna-asnr-miccai brats 2021 benchmark on brain tumor segmentation and radiogenomic classification,” arXiv preprint arXiv:2107.02314, 2021.

3. R. Stupp, W. P. Mason, M. J. van den Bent, M. Weller, B. Fisher, M. J. Taphoorn, K. Belanger, A. A. Brandes, C. Marosi, U. Bogdahn, J. Curschmann, R. C. Janzer, S. K. Ludwin, T. Gorlia, A. Allgeier, D. Lacombe, J. G. Cairncross, E. Eisenhauer,

and R. O. Mirimanof, “Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma,” New England Journal of Medicine, vol. 352, no. 10, pp. 987– 996, 2005.

4. D. LaBella, M. Adewole, M. Alonso-Basanta, T. Altes, S. M. Anwar, U. Baid, T. Bergquist, R. Bhalerao, S. Chen, V. Chung, et al., “The asnr-miccai brain tumor segmentation (brats) challenge 2023: Intracranial meningioma,” 2023.

5. B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, Y. Burren, N. Porz, J. Slotboom, R. Wiest, et al., “The multimodal brain tumor image segmentation benchmark (brats),” IEEE Transactions on Medical Imaging, vol. 34, no. 10, pp. 1993–2024, 2015.

6. S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, R. T. Shinohara, C. Berger, S. M. Ha, M. Rozycki, et al., “Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the brats challenge,” arXiv preprint arXiv:1811.02629, 2018.

7. M. Aboian, U. Anazodo, U. Baid, S. Bakas, E. Calabrese, G. M. Conte, A. Kazerooni, Y. Kirchhof, F. Kofler, D. LaBella, H. Li, M. G. Linguraru, B. Menze, J. Rudie, P. Vollmuth, and B. Wiestler, “MICCAI 2025 Lighthouse Challenge: Brain Tumor Segmentation Cluster of Challenges (BraTS),” Mar. 2025. Challenge proposal, MICCAI 2025.

8. M. J. Sheller, B. Edwards, G. A. Reina, J. Martin, S. Pati, A. Kotrotsou, M. Milchenko, W. Xu, D. Marcus, R. R. Colen, et al., “Federated learning in medicine: facilitating multi-institutional collaborations without sharing patient data,” Scientific reports, vol. 10, no. 1, pp. 1–12, 2020.

9. N. Rieke, J. Hancox, W. Li, F. Milletari, H. R. Roth, S. Albarqouni, S. Bakas, M. N. Galtier, B. A. Landman, K. Maier-Hein, S. Ourselin, M. Sheller, R. M. Summers, A. Trask, D. Xu, M. Baust, and M. J. Cardoso, “The future of digital health with federated learning,” npj Digital Medicine, vol. 3, p. 119, Sept. 2020.

10. S. Pati, U. Baid, B. Edwards, M. Sheller, S.-H. Wang, G. A. Reina, P. Foley, A. Gruzdev, D. Karkada, C. Davatzikos, et al., “Federated learning enables big data for rare cancer boundary detection,” arXiv preprint arXiv:2204.10836, 2022.

11. M. Zenk, U. Baid, S. Pati, A. Linardos, B. Edwards, M. Sheller, P. Foley, A. Aristizabal, D. Zimmerer, A. Gruzdev, et al., “Towards fair decentralized benchmarking of healthcare AI algorithms with the Federated Tumor Segmentation (FeTS) challenge,” Nature Communications, vol. 16, no. 1, p. 6274, 2025.

12. A. Linardos, S. Pati, U. Baid, B. Edwards, P. Foley, K. Ta, V. Chung, M. Sheller, M. I. Khan, M. Jafaritadi, et al., “The miccai federated tumor segmentation (fets) challenge 2024: Eficient and robust aggregation methods for federated learning,” arXiv preprint arXiv:2512.06206, 2025.

13. H. R. Roth, Y. Cheng, Y. Wen, I. Yang, Z. Xu, Y.-T. Hsieh, K. Kersten, A. Harouni, C. Zhao, K. Lu, et al., “Nvidia flare: Federated learning from simulation to realworld,” arXiv preprint arXiv:2210.13291, 2022.

14. S. Bakas, S. Thakur, U. Baid, A. Linardos, S. Pati, J. Doshi, and R. T. Shinohara, “My model is better than yours! statistically-aware ranking for fair benchmarking of ai models,” in Segmentation, Classification, and Synthesis for Brain Tumors and Traumatic Brain Injuries (S. Bakas, E. Dennis, M. Astaraki, U. Baid, G. M. Conte, M. Foltyn-Dumitru, Z. Jiang, D. Labella, M.-C. Metz, U. Anazodo, M. C. de Verdier, F. Kofler, H. B. Li, M. G. Linguraru, and N. Maleki, eds.), (Cham), pp. 3–13, Springer Nature Switzerland, 2026.

15. D. LaBella, O. Khanna, S. McBurney-Lin, et al., “A multi-institutional meningioma MRI dataset for automated multi-sequence image segmentation,” Scientific Data, vol. 11, p. 496, 2024.

16. M. Adewole, J. D. Rudie, A. Gbadamosi, et al., “The BraTS-Africa dataset: Expanding the brain tumor segmentation data to capture African populations,” Radiology: Artificial Intelligence, vol. 7, no. 4, 2025.

17. M. Akhtar, O. Benjelloun, C. Conforti, L. Foschini, P. Gijsbers, J. Giner-Miguelez, S. Goswami, N. Jain, M. Karamousadakis, S. Krishna, et al., “Croissant: A metadata format for ml-ready datasets,” Advances in Neural Information Processing Systems, vol. 37, pp. 82133–82148, 2024.

18. Schema.org, “Schema.org.” https://schema.org, 2026. Accessed: 2026-10-12.

19. R. Cox, J. Ashburner, H. Breman, K. Fissell, C. Haselgrove, C. Holmes, J. Lancaster, D. Rex, S. Smith, J. Woodward, et al., “A (sort of) new image data format standard: Nifti-1: We 150,” Neuroimage, vol. 22, 2004.

20. T. Rohlfing, N. M. Zahr, E. V. Sullivan, and A. Pfeferbaum, “The sri24 multichannel atlas of normal adult human brain structure,” Human brain mapping, vol. 31, no. 5, pp. 798–819, 2010.

21. S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. S. Kirby, J. B. Freymann, K. Farahani, and C. Davatzikos, “Advancing the cancer genome atlas glioma mri collections with expert segmentation labels and radiomic features,” Scientific data, vol. 4, no. 1, pp. 1–13, 2017.

22. S. Thakur, J. Doshi, S. Pati, S. Rathore, C. Sako, M. Bilello, S. M. Ha, G. Shukla, A. Flanders, A. Kotrotsou, et al., “Brain extraction on mri scans in presence of diffuse glioma: Multi-institutional performance evaluation of deep learning methods and robust modality-agnostic training,” NeuroImage, vol. 220, p. 117081, 2020.

23. M. J. Cardoso, W. Li, R. Brown, N. Ma, E. Kerfoot, Y. Wang, B. Murrey, A. Myronenko, C. Zhao, D. Yang, V. Nath, Y. He, Z. Xu, A. Hatamizadeh, A. Myronenko, W. Zhu, Y. Liu, M. Zheng, Y. Tang, I. Yang, M. Zephyr, B. Hashemian, S. Alle, M. Z. Darestani, C. Budd, M. Modat, T. Vercauteren, G. Wang, Y. Li, Y. Hu, Y. Fu, B. Gorman, H. Johnson, B. Genereaux, B. S. Erdal, V. Gupta, A. Diaz-Pinto, A. Dourson, L. Maier-Hein, P. F. Jaeger, M. Baumgartner, J. Kalpathy-Cramer, M. Flores, J. Kirby, L. A. D. Cooper, H. R. Roth, D. Xu, D. Bericat, R. Floca, S. K. Zhou, H. Shuaib, K. Farahani, K. H. Maier-Hein, S. Aylward, P. Dogra, S. Ourselin, and A. Feng, “Monai: An open-source framework for deep learning in healthcare,” 2022.

24. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-eficient learning of deep networks from decentralized data,” in Artificial Intelligence and Statistics, pp. 1273–1282, PMLR, 2017.

25. P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, et al., “Advances and open problems in federated learning,” arXiv preprint arXiv:1912.04977, 2019.

26. A. Myronenko, “3d mri brain tumor segmentation using autoencoder regularization,” in International MICCAI brainlesion workshop, pp. 311–320, Springer, 2018.

27. A. P. Zijdenbos, B. M. Dawant, R. A. Margolin, and A. C. Palmer, “Morphometric analysis of white matter lesions in mr images: method and validation,” IEEE transactions on medical imaging, vol. 13, no. 4, pp. 716–724, 1994.

28. Z. Jiang, A. Fathi Kazerooni, M. Astaraki, S. Bakas, V. Chung, K. Farahani, M. Correia de Verdier, U. Baid, D. LaBella, G. M. Conte, S. You, N. Yordanov, M. S. Aboian, P. Buzduga, E. Grosskopf, F. Kofler, B. Wiestler, B. Menze, R. Y. Huang, F. Batal, S. Varshochi, S. Adib Moradi, N. Sheth, K. Nandolia, M. S´anchez-Monta˜no, J. B. Ware, A. Nabavizadeh, A. Vossough, and M. G. Linguraru, “ Multi-Consortium International Pediatric Brain Tumor Segmentation Challenge

(BraTS-PEDs) 2026: Design and Results ,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2026 Workshops and Challenges, vol. LNCS 17253, Springer Nature Switzerland, 2026.

29. L. Maier-Hein, A. Reinke, P. Godau, M. D. Tizabi, F. Buettner, E. Christodoulou, B. Glocker, F. Isensee, J. Kleesiek, M. Kozubek, M. Reyes, M. A. Riegler, M. Wiesenfarth, A. E. Kavur, C. H. Sudre, M. Baumgartner, M. Eisenmann, D. Heckmann-N¨otzel, T. R¨adsch, L. Acion, M. Antonelli, T. Arbel, S. Bakas, A. Benis, M. B. Blaschko, M. J. Cardoso, V. Cheplygina, B. A. Cimini, G. S. Collins, K. Farahani, L. Ferrer, A. Galdran, B. van Ginneken, R. Haase, D. A. Hashimoto, M. M. Hofman, M. Huisman, P. Jannin, C. E. Kahn, D. Kainmueller, B. Kainz, A. Karargyris, A. Karthikesalingam, F. Kofler, A. Kopp-Schneider, A. Kreshuk, T. Kurc, B. A. Landman, G. Litjens, A. Madani, K. Maier-Hein, A. L. Martel, P. Mattson, E. Meijering, B. Menze, K. G. M. Moons, H. M¨uller, B. Nichyporuk, F. Nickel, J. Petersen, N. Rajpoot, N. Rieke, J. Saez-Rodriguez, C. I. S´anchez, S. Shetty, M. van Smeden, R. M. Summers, A. A. Taha, A. Tiulpin, S. A. Tsaftaris, B. Van Calster, G. Varoquaux, and P. F. J¨ager, “Metrics reloaded: recommendations for image analysis validation,” Nature Methods, vol. 21, p. 195–212, Feb. 2024.