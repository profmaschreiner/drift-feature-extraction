# Using Concept Drift Detectors as Feature Extractors for Sequential Data

Companion code, synthetic datasets and results for the paper *"Using Concept Drift Detectors as Feature Extractors for Sequential Data"*, submitted to **Neurocomputing (Elsevier)**.

The methodology accumulates the alarms of a concept drift detector applied to each variable into one feature per variable, the **drift feature**, for supervised classification of multivariate sequential data. It is evaluated with 9 detectors, on 35 synthetic datasets (20 CaDrift and 15 single-factor) and 16 real datasets from 9 independent sources.

---

## Requirements

```bash
conda env create -f environment.yml
conda activate drift-feature-extraction
```

or

```bash
pip install scikit-learn==1.7.2 capymoa==0.12.0 river==0.22.0 pycatch22==0.4.5 optuna==4.7.0
```

Python 3.13.7. The hardware and software environment is described in Table S-4 of the supplementary material. Regenerating the CaDrift datasets requires a separate environment (`data/synthetic/environment-synthetic.yml`, see `data/synthetic/README.md`).

---

## Repository structure

```
.
├── data/
│   ├── synthetic/        # CaDrift and single-factor datasets and their generators
│   └── real/             # Source links of the real datasets
├── src/
│   ├── experiments/      # Algorithm 1 (scoring.py) and scenarios C0-C3
│   ├── preprocessing/    # Preprocessing of each real dataset
│   └── analysis/         # Statistical tests, controls, sensitivity analysis, figures
├── results/              # Results on the synthetic and real datasets, and figures
├── slurm/                # SLURM job scripts
├── environment.yml
├── CITATION.cff
└── LICENSE
```

Drift alarm positions do not depend on φ_b, so they are detected once and cached, and `src/experiments/scoring.py` recomputes the drift feature for each φ_b tried during the Optuna search (30 evaluations, `seed=42`).

---

## Citation

See [`CITATION.cff`](CITATION.cff). BibTeX will be added upon acceptance.

## License

Code is released under the MIT License (see [`LICENSE`](LICENSE)). Synthetic datasets in `data/synthetic/` are released under CC-BY 4.0.

## Authors

Marcos A. Schreiner, Renan A. N. G. Escribano, Luiz Eduardo S. de Oliveira (Federal University of Paraná, UFPR)
Heitor Gomes (Victoria University of Wellington, VUW)
