# Installation Guide

The environment used for this project was Windows 11 with Python 3.11.9 (the README also lists Python
3.10). These steps describe that environment; they have not been re-run on a fresh machine. Use PowerShell.

## 1. Environment

```powershell
cd medical_sam_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. PyTorch (platform-specific; install first)

```powershell
# CPU only
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
# NVIDIA GPU (CUDA 12.6)
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu126
```

## 3. Other dependencies

```powershell
pip install -r requirements.txt   # needs git (segment-anything is pinned to a commit)
```

The external-evaluation scripts also use `psutil`. It is installed in the project environment
(version 7.1.0) but is **not listed in `requirements.txt`**. Install it separately if it's missing:
`pip install psutil==7.1.0`.

## 4. SAM checkpoint

```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
```

Expected SHA-256: `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`. Check it with
`Get-FileHash checkpoints\sam_vit_b_01ec64.pth`.

## 5. Data

* **MSD Task09 Spleen:** extract `Task09_Spleen.tar` to `data/raw/Task09_Spleen/`, then run
  `python -m src.preprocessing`.
* **TotalSegmentator subset (external):** the archive belongs in
  `..\external_datasets\totalsegmentator_small_v201\archive\`. Verify its MD5 before use. See
  [REPRODUCIBILITY_GUIDE](REPRODUCIBILITY_GUIDE.md).

## 6. Hardware

* **Minimum (CPU):** 4 cores, 8 GB of RAM, and at least 5 GB of free disk space plus the datasets.
* **SAM on CPU:** peak working set 1.44–1.45 GiB (about 1.55 GB) with the external memory optimisation; 3.07 GiB (3.29 GB)
  without it.
* **If RAM is short,** close browsers before long runs.

## 7. Running the tests safely

```powershell
python -B -m pytest -p no:cacheprovider tests
python -B -m pytest -p no:cacheprovider results/external_ts/scripts/tests
```

* `-B` and `-p no:cacheprovider` stop Python and pytest from writing cache files into the project.
* Two tests in `tests/` load the real checkpoint when it is present. They need about 1.5–3 GB of free RAM.
  To exclude them (the project default):

```powershell
python -B -m pytest -p no:cacheprovider tests --deselect tests/test_enhanced_prompts.py::test_real_sam_enhanced_case_006_slice_095 --deselect tests/test_fob.py::test_real_sam_shapes_and_no_dropout
```

* **Counts on 2026-11-23:**
  * `tests/`: 93 passed, 2 deselected;
  * `results/external_ts/scripts/tests` (run from inside that folder with `--rootdir=.`): 108 passed.

## 8. Running the exploratory dashboard (Phase C)

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
.\.venv\Scripts\python.exe -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000
```

1. Wait for `http://127.0.0.1:8000/ext/health` to report `"sam_loaded": true`. SAM needs about 1.5 GiB
   of free RAM.
2. Open `frontend/dashboard.html` in a browser.
3. The app writes only to `app_data/`. Set `MEDSAM_EXT_DATA` to put it elsewhere.

Details are in [EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md).
