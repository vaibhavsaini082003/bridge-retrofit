# Bridge Failure Analysis & Retrofit Recommendation

This project builds an ML workflow for:
- Severity / risk prediction (classification)
- Similar-case retrieval (KNN / optional FAISS)
- Retrofit recommendation (vote from similar cases)

Designed for **Google Colab + Drive** first, but works locally too.

## Recommended workflow (notebooks)

Run notebooks in this order:
1. [notebooks/01_EDA.ipynb](notebooks/01_EDA.ipynb) — quick sanity checks (columns, missingness)
2. [notebooks/02_Preprocessing.ipynb](notebooks/02_Preprocessing.ipynb) — fit preprocessors + save selected dataset
3. [notebooks/03_Severity_Model.ipynb](notebooks/03_Severity_Model.ipynb) — train severity model + demo
4. [notebooks/04_KNN_Similarity.ipynb](notebooks/04_KNN_Similarity.ipynb) — fit similarity index + demo
5. [notebooks/05_Evaluation.ipynb](notebooks/05_Evaluation.ipynb) — evaluation report
6. [notebooks/06_Full_Pipeline_Demo.ipynb](notebooks/06_Full_Pipeline_Demo.ipynb) — combined demo (severity + similarity)

## Data inputs

Default dataset location:
- `data/raw/bridge_data.csv`

Target columns expected by default config:
- Severity: `Severity`
- Retrofit label (for similarity voting): `Suggested_Retrofit`

Change paths/column names in [configs/default.yaml](configs/default.yaml) or use the large preset [configs/colab_large.yaml](configs/colab_large.yaml).

## Artifacts / “timestamp folders” (why they exist)

Every training step writes outputs to a new run folder like:
`artifacts/20260511T195003Z/`

This is intentional:
- It prevents overwriting older models
- It keeps a history of experiments + metrics
- You can compare runs or roll back later

The full pipeline demo and `predict` command **auto-load the latest saved checkpoints** from `artifacts/` (no copies, no extra folders).

## CLI (optional)

All steps can be run via CLI. This is what the notebooks call.

```powershell
python -m bridge_retrofit.cli --config configs/default.yaml preprocess
python -m bridge_retrofit.cli --config configs/default.yaml train --task severity
python -m bridge_retrofit.cli --config configs/default.yaml fit-similarity
python -m bridge_retrofit.cli --config configs/default.yaml evaluate
```

Single-record prediction:

```powershell
python -m bridge_retrofit.cli --config configs/default.yaml predict --json "{\"Bridge_Type\":\"Beam\",\"Material\":\"Steel\",\"Age_Years\":25}"
```

## Git tracking (what’s committed vs ignored)

This repo includes a `.gitignore` that:
- Tracks the folder structure via `.gitkeep`
- Ignores large/volatile outputs (`artifacts/`, `data/raw/`, `data/processed/`, caches)

If you already added CSVs or artifacts to git earlier, remove them from the index once:
```powershell
git rm -r --cached artifacts data\raw data\processed
```
