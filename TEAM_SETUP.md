# Team Setup and Demo: Get the Project Running on Your Own Laptop

There are three levels. Do them in order; each one builds on the previous.

| Level | What you see | Needs | Time |
|---|---|---|---|
| **A. Results demo** | All tests pass, and every published number is recomputed from the raw results | Git, Python, about 2 GB of disk | 15–25 min (mostly downloads) |
| **B. Live dashboard** | SAM segmenting a real spleen CT slice in your browser, with all 4 methods, uncertainty maps and click-to-refine | Level A + the SAM model (375 MB) + the MSD dataset (1.5 GB) + **at least 1.5 GB of free RAM** | +30 min |
| **C. Re-run an evaluation** | The development evaluation (v1/v2 style) on your machine | Level B | 10–60 min on CPU |

The SAM model file and the CT images are **not on GitHub** (too large, and medical data stays out of git). You
download them yourself in Level B.

---

## Level A: results demo (no SAM, no dataset)

### A1. Install the tools (once)
* **Git:** <https://git-scm.com/download/win>
* **Python 3.11** (3.10 also works): <https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"**.

### A2. Download the project from GitHub
Open **PowerShell** (Windows) or **Terminal** (macOS/Linux):

```powershell
git clone https://github.com/ThrashaShetty/Majorr-Project.git
cd Majorr-Project
```

### A3. Create a private Python environment and install packages

Windows (PowerShell):
```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass      # only needed if activation is blocked
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install psutil==7.1.0
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install psutil==7.1.0
```

You should now see `(.venv)` at the start of your prompt. Every command below assumes it is active.

### A4. Run the tests
```powershell
python -B -m pytest -p no:cacheprovider -q tests --deselect tests/test_enhanced_prompts.py::test_real_sam_enhanced_case_006_slice_095 --deselect tests/test_fob.py::test_real_sam_shapes_and_no_dropout
cd results/external_ts/scripts/tests
python -B -m pytest -p no:cacheprovider -q --rootdir=. .
cd ../../../..
```
Expected:
```
93 passed, 2 deselected
110 passed
```

### A5. Run the results demo
```powershell
python -B demo_results.py
```
It recomputes the internal and external statistics from the raw per-slice rows, checks they are identical to the
saved results, and prints the tables. Key lines you should see (condensed):
```
v3 statistics reproduced exactly: YES
primary     identical to the saved analysis.json: YES
sensitivity identical to the saved analysis.json: YES
EXTERNAL PRIMARY COHORT (TotalSegmentator, Basel): 74 patients, 826 slices
Baseline 0.452 · Enhanced v2 0.289 · A5 0.450 · Single-mask 0.288
H1 A5 vs Baseline (PRIMARY)    -0.002   [-0.021, +0.016]   Holm p 1
H3 Enhanced v2 vs Baseline     -0.163   [-0.207, -0.120]   Holm p 4.3e-08
ALL CHECKS PASSED
```

### A6. Look at the figures and reports (no commands needed)
* `results/external_ts/figures/phaseB_20261124T052029Z/`: external box plots and paired-difference plots.
* `results/external_ts/figures/slides_v3/fig5_v3_holdout_dice_box.png`: internal held-out results.
* `results/v3/REPORT.md`: the internal held-out report.
* `docs/FINAL_REPORT_DRAFT.md`: the full report. `docs/VIVA_QA.md`: likely viva questions.

---

## Level B: live dashboard with real SAM

### B1. Download the SAM model (375 MB)
Windows:
```powershell
Invoke-WebRequest https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -OutFile checkpoints\sam_vit_b_01ec64.pth
Get-FileHash checkpoints\sam_vit_b_01ec64.pth
```
macOS/Linux: `curl -L -o checkpoints/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth`

The SHA-256 must be `EC2DF62732614E57411CDCF32A23FFDF28910380D03139EE0F4FCBE91EB8C912`.

### B2. Download and prepare the spleen CT dataset (1.5 GB)
1. Download **Task09_Spleen.tar** from <http://medicaldecathlon.com/> (licence CC-BY-SA 4.0).
2. Extract it so these folders exist:
   ```
   data/raw/Task09_Spleen/imagesTr/
   data/raw/Task09_Spleen/labelsTr/
   ```
   (Windows 10/11 can extract .tar with: `tar -xf Task09_Spleen.tar -C data/raw`)
3. Convert the CT scans into PNG slices:
   ```powershell
   python -m src.preprocessing
   ```
   Expected: 209 slices kept, written to `data/images/` and `data/masks/`.

### B3. Start the backend
Close Chrome and other heavy apps first: SAM needs about **1.5 GB of free RAM**.
```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
python -B -m uvicorn backend.extensions.serve:app --host 127.0.0.1 --port 8000
```
Leave this window open. In a browser, open <http://127.0.0.1:8000/ext/health> and wait until it shows
`"sam_loaded": true` (about 30–60 seconds).

### B4. Use the dashboard
1. Open `frontend/dashboard.html` in your browser (double-click it in File Explorer).
2. Upload a **support** pair and a **query** from a **different patient**. A tested example from the v3 run:
   * Support image: `data/images/case_008_slice_020.png`
   * Support mask: `data/masks/case_008_slice_020.png`
   * Query image: `data/images/case_002_slice_080.png`
   * Query mask (optional, gives the scores): `data/masks/case_002_slice_080.png`
3. Choose a method (Baseline, Enhanced v2, A5 or Single-mask) and click **Segment**. On a CPU laptop each run takes
   roughly 10–30 seconds (longer with uncertainty switched on).
4. You will see:
   * the overlay with the prompt box (yellow) and points (green +, red −);
   * the predicted mask;
   * the uncertainty heatmaps (entropy and variance);
   * Dice, IoU, Precision, Recall and Boundary IoU (when you gave the query mask).

   In the v3 run this query scored Dice 0.91 with Baseline, 0.80 with Enhanced v2 and 0.91 with A5
   (`results/v3/adaptive/overlays/case_002_slice_080.png`). The dashboard is an exploratory tool, so its numbers can
   differ slightly from the frozen evaluation.
5. **Click to refine:** click on the overlay to add a positive point (or switch to negative) and SAM re-segments.
   **Reset clicks** undoes them.
6. **History:** every run is saved; you can compare runs and export CSV, HTML or PDF reports.

To stop the backend, press `Ctrl + C` in its window.

---

## Level C: re-run an evaluation yourself (optional)
```powershell
python run_evaluation.py --max-queries 20 --results-dir results/my_test_run
```
This runs Baseline, FoB and Enhanced on 20 development slices and writes its own CSV and charts into
`results/my_test_run/`. **Always use a new `--results-dir`** so the committed results are never overwritten.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognised | Reinstall Python with "Add to PATH" ticked, or use `py -3.11` instead of `python` |
| `Activate.ps1 cannot be loaded` | Run `Set-ExecutionPolicy -Scope Process Bypass` first |
| `pip install -r requirements.txt` fails on segment-anything | Git must be installed (it downloads SAM's code from GitHub) |
| `/ext/health` shows `sam_loaded: false` | The checkpoint is missing or the wrong file; check B1 and its SHA-256 |
| Backend crashes or freezes while loading | Not enough RAM: close other programs (SAM needs about 1.5 GB free) |
| Dashboard says the backend is unreachable | Start the backend (B3) and keep its window open |
| `No 'imagesTr' directory found` | The dataset is not extracted into `data/raw/Task09_Spleen/` |
