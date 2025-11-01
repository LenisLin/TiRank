# TiRank

<img src="./docs/source/_static/TiRank_white.png" alt="TiRank Logo" width="50%" />

| | |
| :--- | :--- |
| **Full Documentation** | 📖 [**https://tirank.readthedocs.io**](https://tirank.readthedocs.io) |
| **PyPI** | [![PyPI](https://img.shields.io/pypi/v/tirank?style=flat-square)](https://pypi.org/project/TiRank/) |
| **License** | [![License](https://img.shields.io/github/license/LenisLin/TiRank?style=flat-square)](https://github.com/LenisLin/TiRank/blob/main/LICENSE) |
| **Build Status** | [![Documentation Status](https://readthedocs.org/projects/tirank/badge/?version=latest&style=flat-square)](https://tirank.readthedocs.io/en/latest/) |

## Table of Contents

- 📚 [a) Overview](#a-overview)
- 🌟 [b) Features](#b-features)
- ⚙️ [c) Requirements](#c-requirements)
- 🛠️ [d) Installation](#d-installation)
- 🗂️ [e) Configuration (Files & Layout)](#e-configuration-files--layout)
- ▶️ [f) Usage (How to Run)](#f-usage-how-to-run)
- 📊 [g) Output Structure](#g-output-structure)
- ✅ [h) Testing](#h-testing)
- 📚 [Full Documentation](#full-documentation)
- 🧑‍💻 [Support](#support)
- 📜 [License](#license)

---

## a) Overview

**TiRank** integrates bulk RNA‐seq with single-cell or spatial transcriptomics to identify phenotype-associated regions or cell clusters. It supports Cox survival analysis, classification, and regression, and ships with Python scripts and a web GUI.

![TiRank Workflow](./docs/source/_static/Fig1.png)

---

## b) Features

- **🔗 Integration**: Bulk + scRNA-seq or spatial transcriptomics  
- **🔄 Modes**: Cox survival, classification, regression  
- **📈 Visualization**: UMAPs, spatial maps, score distributions  
- **⚙️ Tunable**: Key hyperparameters exposed  
- **🧰 Interfaces**: Scripts/CLI, Python API, Web GUI

---

## c) Requirements

### System
- **OS**: Tested on `Ubuntu 22.04`, `Ubuntu 20.04`
- **Python**: `3.9`
- **RAM**: ≥ **16 GB** (more for large datasets)
- **GPU (important)**: **CUDA-compatible GPU required**
  - Tested with `RTX 2080Ti (CUDA 11.2)`, `RTX 3090 (CUDA 12.1)`, `RTX 4090 (CUDA 12.5)`
- **Disk**: Enough for datasets and intermediate files

### Data
- **Spatial or Single-Cell data**: To characterize heterogeneity  
- **Bulk data**: Expression matrix + clinical metadata aligned by sample IDs

---

## d) Installation

TiRank supports multiple setups.

<!-- ### Method 1: pip (Quick Start)
```bash
pip install tirank
``` -->

### Method 1: Conda (Recommended)
```bash
cd TiRank
conda env create -f ./installation/environment.yml
conda activate Tirank
````

### Method 2: Docker

```bash
docker pull lenislin/tirank_v1:latest
docker run -it --gpus all -p 8050:8050 -v $PWD:/workspace lenislin/tirank_v1:latest /bin/bash
```

### Method 3: Interactive Web Tool (GUI)

See **[GUI Tutorial](https://tirank.readthedocs.io/en/latest/tutorial_web.html)**.

### (Required for examples) Download example data

If you plan to run the example scripts in **f) Usage**, download and place the data now:

* 📥 **Sample data**: [https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE](https://drive.google.com/drive/folders/1CsvNsDOm3GY8slit9Hl29DdpwnOc29bE)
* Unzip/place the folders under `data/ExampleData/` exactly as shown in **e) Configuration**.

---

## e) Configuration (Files & Layout)

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

Edit **two variables** in each script—`dataPath` and `savePath`—then run from the repo root.

### 1) scRNA-seq → bulk (Melanoma response)

**Script:** `Example/SC-Response-SKCM.py`

**Run:**

```bash
python Example/SC-Response-SKCM.py
```

**Tutorial:** [https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html](https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html)

---

### 2) Spatial transcriptomics → bulk (CRC survival)

**Script:** `Example/ST-Cox-CRC.py`

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

### Final output interpretation

The file **`spot_predict_score.csv`** contains **`Rank_Label`**:

* **Cox (survival)**: `TiRank+` → worse survival, `TiRank-` → better survival
* **Classification**: `TiRank+` ↔ phenotype `1`, `TiRank-` ↔ phenotype `0`
* **Regression**: `TiRank+` → higher phenotype score, `TiRank-` → lower phenotype score

Use this for downstream tasks like subpopulation discovery, DEG, and pathway enrichment.

---

## h) Testing

### Quick installation test

```bash
python - <<'PY'
import tirank
print("TiRank version:", getattr(tirank, "__version__", "unknown"))
PY
```

### Full test with example data (single script)

1. **Confirm you already downloaded and placed** the data in **d) Installation**.
2. **Configure** `Example/SC-Response-SKCM.py`:

   ```python
   dataPath = "./data/ExampleData/SKCM_SC_Res"
   savePath = "./results/SC_Respones_SKCM"
   ```
3. **Run**

   ```bash
   python Example/SC-Response-SKCM.py
   ```
4. **Verify**

   * Check `<savePath>/3_Analysis/spot_predict_score.csv`
   * Ensure a valid `Rank_Label` column is present.

---

## Full Documentation

Complete guides, tutorials, API reference, and result interpretation:

### ➡️ **[https://tirank.readthedocs.io](https://tirank.readthedocs.io)**

---

## Support

Questions or issues? Open an issue on GitHub:
[https://github.com/LenisLin/TiRank/issues](https://github.com/LenisLin/TiRank/issues)

---

## License

TiRank is released under the **MIT License**.
