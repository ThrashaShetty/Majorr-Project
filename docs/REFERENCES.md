# References

Every entry below is a real, published work. Documents in `docs/` cite them by key, for example
[Kirillov2023].

**Verification status**, checked on 2026-11-23 against Crossref (`api.crossref.org`), arXiv abstract pages
or the Zenodo record. [Bo2026], [MaierHein2024], [Holm1979] and [Efron1993] were re-checked on 2026-11-24:

* **VERIFIED** — title, authors and venue matched an authoritative record. The DOI shown is the record's DOI.
* **PARTIAL** — only some fields could be confirmed; the gap is stated.
* **UNVERIFIED** — no authoritative record was retrieved. Check it with the publisher before submission.
  No missing detail has been filled in by guesswork.

## Models and architectures

* **[Kirillov2023]** A. Kirillov, E. Mintun, N. Ravi, H. Mao, C. Rolland, L. Gustafson, T. Xiao,
  S. Whitehead, A. C. Berg, W.-Y. Lo, P. Dollár, R. Girshick. "Segment Anything." *2023 IEEE/CVF
  International Conference on Computer Vision (ICCV)*, pp. 3992–4003, 2023. doi:10.1109/ICCV51070.2023.00371.
  arXiv:2304.02643. Code and checkpoints: <https://github.com/facebookresearch/segment-anything>.
  — **VERIFIED** (Crossref, arXiv).
* **[Dosovitskiy2021]** A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner,
  M. Dehghani, M. Minderer, G. Heigold, S. Gelly, J. Uszkoreit, N. Houlsby. "An Image is Worth 16x16
  Words: Transformers for Image Recognition at Scale." *International Conference on Learning
  Representations (ICLR)*, 2021. arXiv:2010.11929.
  — **PARTIAL**: title and authors verified on arXiv. The arXiv page says only "ICLR camera-ready
  version", so the ICLR 2021 venue and year are not confirmed by an authoritative record.
* **[Li2022]** Y. Li, H. Mao, R. Girshick, K. He. "Exploring Plain Vision Transformer Backbones for
  Object Detection." *Computer Vision – ECCV 2022*, Lecture Notes in Computer Science, Springer, Cham,
  pp. 280–296, 2022. doi:10.1007/978-3-031-20077-9_17. arXiv:2203.16527. This is the windowed plus
  global-attention design used by SAM's ViT image encoder.
  — **VERIFIED** (Crossref, arXiv). The LNCS volume number is not stated in the record.
* **[He2022]** K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick. "Masked Autoencoders Are Scalable
  Vision Learners." *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*,
  pp. 15979–15988, 2022. doi:10.1109/CVPR52688.2022.01553. arXiv:2111.06377. This is the MAE pre-training
  of SAM's image encoder.
  — **VERIFIED** (Crossref, arXiv).

## SAM in medical imaging

* **[Ma2024]** J. Ma, Y. He, F. Li, L. Han, C. You, B. Wang. "Segment anything in medical images."
  *Nature Communications* 15, 654 (2024). doi:10.1038/s41467-024-44824-z.
  — **VERIFIED** (Crossref).
* **[Mazurowski2023]** M. A. Mazurowski, H. Dong, H. Gu, J. Yang, N. Konz, Y. Zhang. "Segment anything
  model for medical image analysis: an experimental study." *Medical Image Analysis* 89, 102918 (2023).
  doi:10.1016/j.media.2023.102918.
  — **VERIFIED** (Crossref).
* **[Bo2026]** Y. Bo, Y. Zhu, P. Koniusz, H. Zhang. "Focus on Background: Exploring SAM's Potential in
  Few-shot Medical Image Segmentation with Background-centric Prompting." *Proceedings of the IEEE/CVF
  Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 30032–30041, 2026. arXiv:2603.21287.
  Code: <https://github.com/primebo1/FoB_SAM>.
  — **VERIFIED** (CVF open-access BibTeX, arXiv abstract page). This is the FoB method that inspired this
  project. FoB trains a background-prompt generator. This project's BPPC, BCM and SPR modules are a
  **training-free re-implementation inspired by FoB (Bo et al., CVPR 2026)**, built on frozen SAM ViT-B
  features; they are not the authors' code or trained model.

## Datasets

* **[Antonelli2022]** M. Antonelli et al. "The Medical Segmentation Decathlon." *Nature
  Communications* 13, 4128 (2022). doi:10.1038/s41467-022-30695-9.
  — **PARTIAL**: the DOI resolves to the Nature article URL, but the article metadata could not be read
  (the publisher's page needs a login). Volume, article number and full authors are unconfirmed.
* **[Simpson2019]** A. L. Simpson et al. "A large annotated medical image dataset for the development
  and evaluation of segmentation algorithms." arXiv:1902.09063 (2019). Describes Task09 Spleen: 61
  portal-venous CT scans from Memorial Sloan Kettering Cancer Center, released under CC-BY-SA 4.0.
  — **VERIFIED** (arXiv page and HTML full text: title, Task09 description and licence).
* **[Wasserthal2023]** J. Wasserthal et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic
  Structures in CT Images." *Radiology: Artificial Intelligence* 5(5), e230024 (2023).
  doi:10.1148/ryai.230024.
  — **PARTIAL**: the DOI is confirmed by the official Zenodo dataset record, and the paper's methods
  were read on PubMed Central (PMC10546353). The full author list, volume and issue were not checked
  against the publisher, so the entry is shortened to "et al.".
* **[TS-Zenodo]** "Small subset of TotalSegmentator Dataset", version 2.0.1. Zenodo record 10047263.
  doi:10.5281/zenodo.10047263. Licence: CC BY 4.0.
  — **VERIFIED** (Zenodo API).

## Metrics and statistics

* **[Dice1945]** L. R. Dice. "Measures of the Amount of Ecologic Association Between Species."
  *Ecology* 26(3), 297–302 (1945). doi:10.2307/1932409.
  — **VERIFIED** (Crossref).
* **[Cheng2021]** B. Cheng, R. Girshick, P. Dollár, A. C. Berg, A. Kirillov. "Boundary IoU: Improving
  Object-Centric Image Segmentation Evaluation." *2021 IEEE/CVF Conference on Computer Vision and Pattern
  Recognition (CVPR)*, pp. 15329–15337, 2021. doi:10.1109/CVPR46437.2021.01508. arXiv:2103.16562.
  — **VERIFIED** (Crossref, arXiv).
* **[Taha2015]** A. A. Taha, A. Hanbury. "Metrics for evaluating 3D medical image segmentation:
  analysis, selection, and tool." *BMC Medical Imaging* 15, 29 (2015). doi:10.1186/s12880-015-0068-x.
  — **VERIFIED** (Crossref).
* **[MaierHein2024]** L. Maier-Hein et al. "Metrics reloaded: recommendations for image analysis
  validation." *Nature Methods* 21(2), 195–212 (2024). doi:10.1038/s41592-023-02151-z.
  — **VERIFIED** (Crossref DOI record; the DOI resolves to nature.com): title, journal, volume 21, issue 2,
  pages 195–212, February 2024, first author Maier-Hein (73 authors).
* **[Wilcoxon1945]** F. Wilcoxon. "Individual Comparisons by Ranking Methods." *Biometrics Bulletin*
  1(6), 80–83 (1945). doi:10.2307/3001968.
  — **VERIFIED** (Crossref) for title, author, volume, issue, year, start page 80 and DOI. The end page
  (83) is not stated in the record.
* **[Holm1979]** S. Holm. "A Simple Sequentially Rejective Multiple Test Procedure." *Scandinavian
  Journal of Statistics* 6(2), 65–70 (1979). JSTOR 4615733.
  — **VERIFIED** (the journal's scanned article with its JSTOR stable ID, and bibliographic records).
  **No DOI exists** for this article; Crossref has no record of it.
* **[Efron1993]** B. Efron, R. J. Tibshirani. *An Introduction to the Bootstrap.* Monographs on
  Statistics and Applied Probability 57. Chapman & Hall, New York, 1993. ISBN 0-412-04231-2.
  — **VERIFIED** (library catalogue records and the publisher's listing for this ISBN). This is the single
  edition of the book.

## Software

* PyTorch 2.10 (CPU), segment-anything (commit `dca509fe793f601edb92606367a655c15ac00fdf`), NumPy 2.2.6,
  SciPy 1.16.2, OpenCV 4.12, nibabel 5.3.2, pandas 2.3.3, FastAPI 0.115.12.

## Open items before submission

1. **PARTIAL entries:** [Dosovitskiy2021] venue, [Antonelli2022] metadata, [Wasserthal2023] full author
   list.

Resolved on 2026-11-24: the FoB citation ([Bo2026]); [Efron1993] (Chapman & Hall 1993); [MaierHein2024]
(DOI and issue added); [Holm1979] (verified, no DOI exists).
