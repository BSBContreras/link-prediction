# Link prediction (co-authorship)

The implementation and experiments for this project live in the **`method/`** directory, in the Jupyter notebook:

- [`method/link_prediction_resultados_artigo.ipynb`](method/link_prediction_resultados_artigo.ipynb)

## What is implemented

The notebook builds a **co-authorship graph** from bibliographic CSV data (`authorships.csv`, `works.csv`) and evaluates **author-to-author recommendation** (who to collaborate with next) under a train / test split.

**Models**

- **Topology (graph coauthor):** recommends authors via the co-authorship network (e.g. candidates among neighbors-of-neighbors), with a popularity-based fallback.
- **Ideal topology (oracle):** reorders a large candidate list from the topology model so that known future coauthors appear first—an upper bound on ranking quality for that candidate pool.
- **Hybrid (graph + Random Forest):** scores candidate pairs using **link-prediction-style features** (common neighbors, Jaccard similarity, Adamic–Adar) and trains a **Random Forest** classifier to rank recommendations.

**Evaluation**

Metrics reported include **Precision, Recall, F1, NDCG@k, and MRR@k** for multiple cutoff values *k*, plus comparison plots across models.

## Requirements

Python with common scientific stack (e.g. NumPy, pandas, scikit-learn, matplotlib, tqdm). See `.python-version` for the intended interpreter version. Place the expected CSV dataset where the notebook’s `database_path` points before running cells that load data.
