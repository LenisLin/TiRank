# Model Input

This page summarizes the expected **input formats** for TiRank and provides practical guidance for matching
example scripts, the Snakemake workflow, and your own datasets.

## Supported analysis modes

TiRank supports three primary modes (driven by the bulk phenotype definition):

- **Cox (survival analysis)**: time + event indicators
- **Classification**: binary or multi-class labels (commonly 0/1)
- **Regression**: continuous phenotype score

Inference data can be **spatial transcriptomics (ST)** or **single-cell RNA-seq (SC)**.

---

## 1) Bulk RNA-seq expression matrix

**Format**: CSV/TSV (recommended), readable by pandas.

**Recommended orientation**:
- **Rows = genes**
- **Columns = samples**

**Requirements**:
- Gene identifiers should be consistent (e.g., HGNC gene symbols) and match across datasets where applicable.
- Sample IDs must match those used in the bulk clinical table.

**Example files (Zenodo example resources)**:
- `GSE39582_exp_os.csv` (bulk expression)

---

## 2) Bulk clinical / phenotype table

**Format**: CSV/TSV.

**Requirements**:
- A sample identifier column that matches the bulk expression column names.
- Columns required depend on mode:

### Cox (survival)
Minimum required columns:
- `sample_id`
- `time` (numeric; follow-up time)
- `event` (0/1; 1 = event occurred)

Example file:
- `GSE39582_clinical_os.csv`

### Classification
Minimum required columns:
- `sample_id`
- `label` (e.g., 0/1)

Example files:
- `Liu2019_meta.csv` (metadata / labels)

### Regression
Minimum required columns:
- `sample_id`
- `score` (numeric phenotype)

---

## 3) Spatial transcriptomics (ST) input

TiRank supports common ST data representations used in Python pipelines.

### A) Visium-style folder input
A directory containing standard Visium outputs (e.g., matrix + spatial metadata).
In the TiRank examples, the ST input is provided as a **folder**:

- `SN048_A121573_Rep1/` (example ST folder)

Example placement (recommended):
- `data/ExampleData/CRC_ST_Prog/SN048_A121573_Rep1/`

### B) AnnData (optional)
If you already have an `.h5ad` AnnData object for ST, you may adapt the example scripts accordingly.

---

## 4) Single-cell RNA-seq (SC) input

**Format**: AnnData `.h5ad` (recommended).

**Requirements** (typical):
- Expression stored in `X` (cells × genes)
- Cell-level metadata stored in `obs` (e.g., patient/sample identifiers and optional covariates)

Example file:
- `GSE120575.h5ad`

Recommended placement:
- `data/ExampleData/SKCM_SC_Res/GSE120575.h5ad`

---

## 5) Pretrained model files (if required by your workflow)

Some workflows require pretrained files such as `ctranspath.pth`.

Recommended placement for CLI/workflow:
- `data/pretrainModel/ctranspath.pth`

Recommended placement for Web GUI:
- `Web/data/pretrainModel/ctranspath.pth`

The example resources are hosted on Zenodo:
- https://zenodo.org/records/18275554

---

## 6) Example resources (recommended starting point)

We provide example datasets and pretrained assets on Zenodo for reproducible testing:

- https://zenodo.org/records/18275554

A recommended local structure:

```
TiRank/
├── data/
│ ├── pretrainModel/
│ │ └── ctranspath.pth
│ └── ExampleData/
│ ├── CRC_ST_Prog/
│ └── SKCM_SC_Res/
└── workflow/
├── Snakefile
└── config/config.yaml
```

If you use different locations, update the paths in the example scripts or in `workflow/config/config.yaml`.