# Architecture Diagram

The whole system, from raw CT to the dashboard and its exports.

* **Solid arrows:** the frozen research pipeline, used for every reported result.
* **Dashed arrows:** the exploratory Phase C extensions. They are new files only, and none of them feeds
  back into the frozen evaluation.

The diagram renders on GitHub, in VS Code (with a Mermaid extension) and at mermaid.live.

```mermaid
flowchart TD
    %% ---------- data ----------
    subgraph DATA["Data"]
        MSD["MSD Task09 Spleen<br/>41 labelled CT volumes"]
        TS["TotalSegmentator v2.0.1 subset<br/>91 eligible subjects (MD5-verified ZIP)"]
    end

    %% ---------- preprocessing ----------
    subgraph PRE["Preprocessing · src/preprocessing.py (frozen)"]
        P1["Every 5th axial slice<br/>(external sensitivity: every 17th)"]
        P2["HU window [-200, 300] → uint8<br/>rot90, resize 512×512"]
        P3["Exclusion rules<br/>(≥ 50 spleen px, binary, contrast)"]
        P1 --> P2 --> P3
    end
    MSD --> P1
    TS -- "streamed from ZIP<br/>prep_external_ts.py + exact QC" --> P1

    P3 --> PAIR["Support pairing<br/>nearest 64×64 thumbnail cosine<br/>(different patient)"]

    %% ---------- SAM ----------
    subgraph SAM["SamEngine · src/baseline.py (frozen)"]
        ENC["SAM ViT-B image encoder<br/>1024² → 256×64×64 embedding<br/>(cached; mem_opt row-chunked attention<br/>for the external run, bit-identical)"]
        DEC["Prompt encoder + mask decoder<br/>3 masks + predicted IoU"]
        ENC --> DEC
    end
    PAIR --> ENC

    %% ---------- methods ----------
    subgraph METH["The 4 evaluated methods (frozen)"]
        BASE["Baseline<br/>support-mask box"]
        subgraph ENHG["Enhanced v2 · src/fob/"]
            BPPC["BPPC<br/>background prototype"] --> BCM["BCM<br/>Background-centric<br/>Context Modeling<br/>attention map"] --> SPR["SPR<br/>negatives + 1 positive<br/>(body mask, seed 42)"]
        end
        A5["A5 selection rule<br/>ratio = Baseline px / support px<br/>0.5 ≤ ratio ≤ 2 → Baseline<br/>else → Enhanced v2"]
        SINGLE["Single-mask<br/>Enhanced v2 prompts,<br/>multimask_output = False"]
    end
    DEC --> BASE
    DEC --> ENHG
    BASE --> A5
    ENHG --> A5
    ENHG --> SINGLE

    %% ---------- metrics & stats ----------
    MET["Metrics · src/metrics.py<br/>Dice, IoU, Precision, Recall, Boundary IoU"]
    BASE --> MET
    ENHG --> MET
    A5 --> MET
    SINGLE --> MET

    subgraph EVAL["Evaluation and statistics"]
        V3["v3 held-out MSD<br/>33 patients / 156 slices"]
        EXT["External TotalSegmentator<br/>frozen run_config + worklist<br/>write-once results, verify, backup<br/>(done: 90/90 verified)"]
        SAP["Frozen SAP v1 + Amendment 01<br/>patient-level Wilcoxon, Holm,<br/>cluster bootstrap 10k (seed 42)<br/>(done)"]
        V3 --> SAP
        EXT --> SAP
    end
    MET --> V3
    MET --> EXT

    %% ---------- exploratory extensions ----------
    subgraph XT["Exploratory extensions (Phase C, new files only)"]
        UNC["Uncertainty · src/extensions/uncertainty.py<br/>multimask entropy / disagreement<br/>K-perturbation variance → heatmap + score"]
        REF["Refinement · src/extensions/refinement.py<br/>mask_input + clicks<br/>simulated user (largest error region)"]
        V3X["v3 exploratory study<br/>experiments/exploratory/v3_uncertainty_and_clicks.py<br/>[PENDING: run]"]
    end
    DEC -. "same predictor" .-> UNC
    DEC -. "same predictor" .-> REF
    UNC -.-> V3X
    REF -.-> V3X

    %% ---------- application ----------
    subgraph APP["Application"]
        API0["FastAPI · backend/main.py (unchanged)<br/>/health /segment /results /compare"]
        APIX["Extended API · backend/extensions/<br/>/ext/segment (4 methods + prompts + uncertainty)<br/>/ext/refine /ext/reset"]
        HIST["History · history.py<br/>SQLite ext_runs (app_data/)"]
        EXP["Export · report.py<br/>CSV · HTML · PDF"]
        UI0["frontend/index.html (unchanged)"]
        UIX["frontend/dashboard.html<br/>inputs, prompts drawn, mask, overlay,<br/>uncertainty, metrics, click-to-refine, history"]
        APIX --> HIST --> EXP
        API0 --- UI0
        APIX --- UIX
    end
    METH -. "frozen classes reused" .-> APIX
    UNC -.-> APIX
    REF -.-> APIX
    API0 -. "wrapped unchanged" .-> APIX
```

## Reading guide

| Layer | Frozen? | Where documented |
|---|---|---|
| Preprocessing, SamEngine, the 4 methods, metrics | Yes. Code hashes are in `results/external_ts/eval/run_config.json` | [METHODOLOGY](METHODOLOGY.md), [SYSTEM_ARCHITECTURE](SYSTEM_ARCHITECTURE.md) |
| External pipeline (manifest, QC, runner, mem_opt, guard, verify, backup) | Yes (runner and config) | [SYSTEM_ARCHITECTURE §3](SYSTEM_ARCHITECTURE.md) |
| Statistics | Yes: SAP v1 plus Amendment 01 | `results/external_ts/analysis_plan/`, [SAP_AMENDMENT_01](SAP_AMENDMENT_01_2026-11-23.md) |
| Uncertainty, refinement, extended API, dashboard, history and export | No. Exploratory, new files | [EXPLORATORY_FEATURES](EXPLORATORY_FEATURES.md), [API_DOCUMENTATION](API_DOCUMENTATION.md) |
