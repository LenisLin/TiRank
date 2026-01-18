# TiRank

<img src="./docs/source/_static/TiRank_white.png" alt="TiRank Logo" width="50%" />

| | |
| :--- | :--- |
| **Full Documentation** | 📖 [**https://tirank.readthedocs.io**](https://tirank.readthedocs.io) |
| **Bioconda** | [![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat)](http://bioconda.github.io/recipes/tirank/README.html) |
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

**TiRank** integrates bulk RNA‐seq with single-cell or spatial transcriptomics to identify phenotype-associated regions or cell clusters. It supports Cox survival analysis, classification, and regression, and ships with standalone Python scripts, a standardized Snakemake workflow, and a user-friendly web GUI.

![TiRank Workflow](./docs/source/_static/Fig1.png)

---

## b) Features

- **🔗 Integration**: Bulk + scRNA-seq or spatial transcriptomics
- **🔄 Modes**: Cox survival, classification, regression
- **📈 Visualization**: UMAPs, spatial maps, score distributions
- **⚙️ Tunable**: Key hyperparameters exposed
- **🧰 Interfaces**: Scripts/CLI, Snakemake Workflow, Python API, Web GUI

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

### Method 1: Bioconda (Recommended)

1. **Clone the repository:**
(Required to access example scripts and the Snakemake workflow)
```bash
git clone [https://github.com/LenisLin/TiRank.git](https://github.com/LenisLin/TiRank.git)
cd TiRank
```

1. **Create and activate the environment:**
```bash
conda create -n tirank python=3.9
conda activate tirank
```

1.  **Install TiRank:**
```bash
conda install -c bioconda tirank
```


### Method 2: Docker

```bash
docker pull lenislin/tirank_v1:latest
docker run -it --gpus all -p 8050:8050 -v $PWD:/workspace lenislin/tirank_v1:latest /bin/bash

```

### Method 3: Interactive Web Tool (GUI)

See **[GUI Tutorial](https://tirank.readthedocs.io/en/latest/tutorial_web.html)**.

**TiRank GUI Video**

<p align="center">
<a href="https://www.youtube.com/watch?v=YMflTzJF6s8">
<img src="docs/source/_static/TiRank_Youtub.png" alt="Watch the video" width="400">
</a>
</p>

### (Required for examples) Download example data

If you plan to run the example scripts in **f) Usage**, please download the testing datasets:

* 📥 **Sample data (Zenodo)**: [18275554](https://zenodo.org/records/18275554)
* Unzip and place the folders under `data/ExampleData/` exactly as shown in **e) Configuration**.

---

## e) Configuration (Files & Layout)

This section shows **where files live** and **which scripts exist**. (How to run is in [f) Usage](https://www.google.com/search?q=%23f-usage-how-to-run).)

### CLI & Workflow

```
TiRank/
├── tirank/                       # CLI entry point for workflows
│   └── tirank_cli.py
├── workflow/                     # Snakemake workflow files
│   ├── Snakefile
│   └── config.yaml
├── Example/
│   ├── SC-Response-SKCM.py       # scRNA-seq → bulk (melanoma response)
│   └── ST-Cox-CRC.py             # Spatial transcriptomics → bulk (CRC survival)
├── data/
│   └── ExampleData/
│       ├── SKCM_SC_Res/
│       │   ├── GSE120575.h5ad    # scRNA-seq datatirank_cli.py
│       │   ├── Liu2019_meta.csv
│       │   └── Liu2019_exp.csv
│       └── CRC_ST_Prog/
│           ├── GSE39582_clinical_os.csv
│           ├── GSE39582_exp_os.csv
│           └── SN048_A121573_Rep1/   # ST folder
└── Example/                      # default output location

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
│       ├── CRC_ST_Prog/
│       └── SKCM_SC_Res/
├── tiRankWeb/
└── app.py

```

> Tip: Keep filenames exactly as shown. Scripts create needed subfolders under `results/` automatically.

---

## f) Usage (How to Run)

You can run TiRank using standalone Python scripts or the standardized Snakemake workflow.

### Option A: Python Scripts

(Optional) Edit **two variables** in each script—`dataPath` and `savePath`—then run from the repo root.

#### 1) scRNA-seq → bulk (Melanoma response)

**Script:** `Example/SC-Response-SKCM.py`
**Data:** `data/ExampleData/SKCM_SC_Res`

**Run:**

```bash
python Example/SC-Response-SKCM.py

```

**Tutorial:** [https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html](https://tirank.readthedocs.io/en/latest/tutorial_sc_classification.html)

#### 2) Spatial transcriptomics → bulk (CRC survival)

**Script:** `Example/ST-Cox-CRC.py`
**Data:** `data/ExampleData/CRC_ST_Prog`

**Run:**

```bash
python Example/ST-Cox-CRC.py

```

**Tutorial:** [https://tirank.readthedocs.io/en/latest/tutorial_st_survival.html](https://tirank.readthedocs.io/en/latest/tutorial_st_survival.html)

---

### Option B: Standardized Workflow (Snakemake)

For reproducible, automated runs on new datasets without modifying code.

1. **Configure**: Edit `workflow/config.yaml` to point to your data paths and set parameters.
2. **Run**:
```bash
cd workflow
snakemake --use-conda -c1

```


*(Replace `-c1` with the number of CPU cores available, e.g., `-c16`)*

---

### Notes

* Use `os.path.join(...)` for portability in custom scripts.
* Ensure `Example/` is writable.

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

### Workflow test (Snakemake)

Verify pipeline integrity and configuration without running full analysis:

```bash
cd workflow
snakemake -n

```

*(If successfully configured, this will print the list of jobs to be executed)*

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