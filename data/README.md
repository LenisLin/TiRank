## e) Files & Layout

This section shows **where files live** and **which scripts exist**. (How to run is in [f) Usage](#f-usage-how-to-run).)

### CLI

```
TiRank/
├── Example/
│   ├── SC-Response-SKCM.py      # scRNA-seq → bulk (melanoma response)
│   └── ST-Cox-CRC.py            # Spatial transcriptomics → bulk (CRC survival)
├── data/
│   └── ExampleData/
│       ├── SKCM_SC_Res/
│       │   ├── Liu2019_meta.csv
│       │   └── Liu2019_exp.csv
│       └── CRC_ST_Prog/
│           ├── GSE39582_clinical_os.csv
│           ├── GSE39582_exp_os.csv
│           └── SN048_A121573_Rep1/   # ST folder (contents as provided)
└── results/                          # any writable location you choose
    ├── SC_Respones_SKCM/
    └── ST_Survival_CRC/
```

### Web GUI

```
Web/
├── assets/
├── components/
├── img/
├── layout/
├── data/
│   ├── pretrainModel/
│   │   └── ctranspath.pth
│   ├── ExampleData/
│   │   ├── CRC_ST_Prog/
│   │   └── SKCM_SC_Res/
├── tiRankWeb/
└── app.py
```

> Tip: Keep filenames exactly as shown. Scripts create needed subfolders under `results/` automatically.

---

## f) Usage (How to Run)

(Optional) Edit **two variables** in each script—`dataPath` and `savePath`—then run from the repo root.

### 1) scRNA-seq → bulk (Melanoma response)

**Script:** `Example/SC-Response-SKCM.py`
**Data:** `data/ExampleData/SKCM_SC_Res`

**Run:**

```bash
python Example/SC-Response-SKCM.py
```

**Tutorial:** [https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html](https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html)

---

### 2) Spatial transcriptomics → bulk (CRC survival)

**Script:** `Example/ST-Cox-CRC.py`
**Data:** `data/ExampleData/CRC_ST_Prog`

**Run:**

```bash
python Example/ST-Cox-CRC.py
```

**Tutorial:** [https://tirank.readthedocs.io/en/latest/tutorial_st_survival.html](https://tirank.readthedocs.io/en/latest/tutorial_st_survival.html)

---

### Notes

* Use `os.path.join(...)` for portability (avoid the typo `os.path.os.path.join`).
* Relative paths above work if you run from the repo root. Absolute paths are fine.
* Ensure `results/` is writable.

---

## g) Output Structure

Each run writes to `<savePath>/` with a consistent structure:

```
<savePath>/
├── 1_loaddata/
│   ├── anndata.pkl
│   ├── bulk_clinical.pkl
│   └── bulk_exp.pkl
│
├── 2_preprocessing/
│   ├── 'bulk gene pair heatmap.png'
│   ├── qc_violins.png
│   ├── scAnndata.pkl
│   ├── sc_gene_pairs_mat.pkl
│   ├── similarity_df.pkl
│   ├── split_data/
│   ├── train_bulk_gene_pairs_mat.pkl
│   └── val_bulkExp_gene_pairs_mat.pkl
│
└── 3_Analysis/
    ├── checkpoints/
    ├── data2train/
    ├── final_anndata.h5ad
    ├── model_para.pkl
    ├── saveDF_bulk.pkl
    ├── saveDF_sc.pkl
    ├── 'TiRank Pred Score Distribution.png'
    ├── 'UMAP of TiRank Label Score.png'
    ├── 'UMAP of TiRank Pred Score.png'
    └── spot_predict_score.csv   <-- **FINAL RESULT FILE**
```