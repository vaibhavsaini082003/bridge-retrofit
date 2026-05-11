# Colab quickstart

1) Mount Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

2) `cd` into the repo folder and install:

```python
%cd /content/drive/MyDrive/retrofit
!pip install -r requirements-colab.txt
!pip install -e .
```

Recommended config for ~100k rows: `configs/colab_large.yaml`.

3) Run the pipeline:

```python
!python -m bridge_retrofit.cli --config configs/colab_large.yaml --project-root /content/drive/MyDrive/retrofit preprocess
!python -m bridge_retrofit.cli --config configs/colab_large.yaml --project-root /content/drive/MyDrive/retrofit train --task severity
!python -m bridge_retrofit.cli --config configs/colab_large.yaml --project-root /content/drive/MyDrive/retrofit fit-similarity
!python -m bridge_retrofit.cli --config configs/colab_large.yaml --project-root /content/drive/MyDrive/retrofit evaluate
```

4) Optional: run the demo UI:

```python
!python -m bridge_retrofit.cli --config configs/colab_large.yaml --project-root /content/drive/MyDrive/retrofit serve --share
```

## Notebook order

If you're running notebook-first, use this order:

- `01_EDA.ipynb`
- `02_Preprocessing.ipynb`
- `03_Severity_Model.ipynb`
- `04_KNN_Similarity.ipynb`
- `05_Evaluation.ipynb`
- `06_Full_Pipeline_Demo.ipynb`
