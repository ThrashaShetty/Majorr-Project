# Dataset Documentation

## 1. MSD Task09 Spleen (development and internal validation)

| Item | Value |
|---|---|
| Source | Medical Segmentation Decathlon [Antonelli2022, Simpson2019], <http://medicaldecathlon.com/> |
| Origin (per [Simpson2019]) | 61 portal-venous CT scans of patients having chemotherapy for liver metastases, Memorial Sloan Kettering Cancer Center |
| Licence | CC-BY-SA 4.0 |
| Used here | the **41 labelled training cases** (`data/raw/Task09_Spleen/imagesTr`, `labelsTr`); the 20 test cases have no public labels |
| Geometry (measured) | 512×512 in-plane; pixel size 0.61–0.98 mm; slice thickness 1.5–8 mm (24 of 41 cases at 5 mm); RAS orientation (array axes point to patient Right, Anterior, Superior) |
| Spleen volume (measured) | 56–502 mL, median about 210 mL |

**Preprocessing result** (`data/preprocessing_report.json`):

* 745 slices sampled (every 5th);
* **209 kept**, 536 excluded, all for `no_foreground` (fewer than 50 spleen pixels).

**Patient-level split** (`data/split_summary.json`, seed 42):

| Set | Patients | Slices |
|---|---|---|
| Support / v3 held-out test | 33 | 156 |
| Query / development | 8 (case_003, 006, 014, 020, 021, 052, 059, 062) | 53 |

No patient appears on both sides.

## 2. TotalSegmentator subset (external evaluation)

| Item | Value |
|---|---|
| Source | [TS-Zenodo] record 10047263, version 2.0.1. Paper [Wasserthal2023] |
| Licence | CC BY 4.0 |
| Archive | `Totalsegmentator_dataset_small_v201.zip`, 3,244,617,817 bytes, MD5 `6b5524af4b15e6ba06ef2d700c0c73e0` (verified) |
| Origin | University Hospital Basel, routine clinical CT; varied scanners, phases and pathologies. No overlap with the MSD institution |
| Labels | Model-assisted, then manually reviewed and corrected [Wasserthal2023]; no inter-rater agreement reported |
| Geometry (measured) | Resampled to 1.5 mm isotropic, RAS. In-plane size 140–333 voxels across all 102 archive subjects; 207–333 voxels across the 91 eligible subjects. Among the 91 eligible subjects, one (s1389) is non-square and tilted 4.4° |
| HU (Hounsfield units) | Stored as int16 with slope 1 and intercept 0 in all 91 subjects; air peaks between −1,024 and −966 |

**Cohort** (frozen manifest, SHA-256 `b6de738e…`):

* 102 subjects in the archive;
* 92 with a non-empty spleen mask;
* **91 eligible** (at least one every-5th slice with ≥ 50 px after resizing);
* 17 of them boundary-touching (spleen partly outside the scan), excluded from the primary cohort.

| Cohort | Subjects | Evaluable subjects | Slices |
|---|---|---|---|
| Primary (every 5th slice, no boundary-touching spleens) | 74 | 74 | 826 |
| Sensitivity (every 17th slice, all eligible) | 91 | 90 (s1358 has none) | 300 |
| Unique slices evaluated (48 in both) | — | — | 1,078 |

**Preprocessing:**

* 1,239 image/mask pairs, all recomputed exactly by independent QC;
* CTs streamed from the ZIP and never extracted;
* 161 boundary-subject every-5th slices are preprocessed but excluded from both analyses.

## 3. Ethics and data handling

* Both datasets are public and de-identified, released under open licences.
* No private or identifiable patient data is used.
* The dashboard must not be used with identifiable data.
