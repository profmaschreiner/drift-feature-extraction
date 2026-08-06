# Using Concept Drift Detectors as Feature Extractors for Sequential Data

Companion code and synthetic datasets for the paper *"Using Concept Drift Detectors as Feature Extractors for Sequential Data"*, submitted to **Knowledge-Based Systems (Elsevier)**.

The methodology transforms **concept drift detector alarms** into compact, interpretable, per-variable features (2*n*-dimensional, up to 11× smaller than *catch22*) for supervised classification of multivariate sequential data. Evaluated with **9 detectors** across **4 experimental scenarios**, on **20 synthetic** (CaDrift) and **16 real** datasets spanning HAR, industrial and environmental domains.

The protocol addresses three points relevant to reproducing the reported statistics:

1. **Pseudoreplication in the real-dataset comparisons** — WARD, PAMAP2, Smartphone-HAR, and Smartwatch-HAR share subjects/sessions/fold structure across their per-sensor splits. Wilcoxon signed-rank tests are paired at the level of **9 independent real-world sources**, not the 16 dataset splits, following Demšar (2006).
2. **Effect size alongside significance** — every comparison reports Cliff's δ with Benjamini-Hochberg correction, not raw p-values alone (thresholds per Meissel & Yao, 2024).
3. **Block cross-validation** — folds follow the natural experimental units of each dataset (subjects, cycles, experiments), with a leave-one-block-out protocol; normalization is fit exclusively on the training block.

---

## 1. Requirements

This repository uses **two separate environments**:

- **`environment.yml`** (this folder) — the main experimental pipeline (detectors, classifiers, statistical analysis). **Python 3.13.7**, versions match Table S-III of the supplementary material.
- **`data/synthetic/environment-synthetic.yml`** — only needed to regenerate the synthetic datasets from scratch via CaDrift. See `data/synthetic/README.md`. Not required if you only use the synthetic datasets already provided in `data/synthetic/`.

### Install (main pipeline)

```bash
conda env create -f environment.yml
conda activate drift-feature-extraction
```

Alternatively, install directly via pip:

```bash
pip install scikit-learn==1.7.2 capymoa==0.12.0 river==0.22.0 pycatch22==0.4.5 optuna==4.7.0
```

---

## 2. Data

### Synthetic (`data/synthetic/`)

The 20 CaDrift generated datasets used in Section 6.1 are included in full and released under CC-BY 4.0. CaDrift (Barboza et al., 2026) is used as a generation dependency and is not vendored in this repository. See `data/synthetic/README.md` for the folder naming convention, the generation environment, and citation.

### Real (`data/real/`)

data/real/README.md` lists the source of each dataset and the preprocessing steps (downsampling, zero-variance removal, per-sensor splitting, Idle-class removal in MH) needed to reconstruct the structure reported in Table 3 of the paper.

---

## 3. Repository structure```
.
├── data/
│   ├── synthetic/
│   │   ├── README.md               # CaDrift dependency and folder naming
│   │   ├── environment-synthetic.yml
│   │   ├── synthetic.py
│   │   └── {Frequency}_{Type}_{Feature}_/   # B1-B20, one folder per scenario
│   └── real/
│       └── README.md               # Source links 
├── src/
│   ├── preprocessing/      # Per-dataset preprocessing (Section 5.1)
│   ├── experiments/        # Scenario runners (C0-C3), Optuna search (Section 5.3)
│   └── analysis/           # Statistical tests, feature importance and extraction cost
├── results/
│   ├── real_results/        # Results obtained on real-world datasets
│   ├── synthetic_results/   # Results obtained on synthetic datasets
│   └── figures/             # Scripts reproducing Figures 5-10 and S-1 to S-7
├── slurm/                   # C3HPC SLURM job scripts
├── environment.yml
├── CITATION.cff
└── LICENSE
```

---

## 4. Reproducibility

- Optuna hyperparameter search: 30 evaluations per scenario/detector, `seed=42`.
- Hardware/software environment used to produce the reported results: Table S-III of the supplementary material (C3HPC cluster, Intel Xeon E5-2683 v4, 16 cores / 16 GB per job).
- Package versions for the main pipeline are pinned in `environment.yml`; synthetic dataset generation uses `data/synthetic/environment-synthetic.yml` (see `data/synthetic/README.md`).

---

## 6. Citation

See [`CITATION.cff`](CITATION.cff). BibTeX will be added upon acceptance.

---

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Synthetic datasets in `data/synthetic/` are released under CC-BY 4.0.

## Authors

Marcos A. Schreiner, Renan A. N. G. Escribano, Luiz Eduardo S. de Oliveira — Federal University of Paraná (UFPR)
Heitor Gomes — Victoria University of Wellington (VuW)
