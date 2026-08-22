# Grand Paris Territorial Typology & Business-Birth Forecasting

**A proof-of-concept for the HARMONIC PhD proposal (CY Cergy Paris Université, PEPR MOBIDEC) — testing whether open French data and machine learning can recover the kind of territorial "urban offer" trajectories that classical LUTI (Land Use–Transport Interaction) models are built to describe.**

> Full proposal: *Modelling Longitudinal Territorial Evolution and Lifestyle Changes: Can Big Data and AI Challenge Classical LUTI Approaches?* — submitted to Mme Liu Liu, August 2026. A summary of how this code maps onto the proposal is in [`docs/proposal_context.md`](docs/proposal_context.md).

## Why this exists

The HARMONIC call asks whether AI can complement or challenge LUTI models in describing how jobs, housing, and other "urban offers" evolve across a territory, and in forecasting territorial change. Before writing the proposal, I built a small, fully open-data pipeline to stress-test that question on the Métropole du Grand Paris (Paris + Hauts-de-Seine, Seine-Saint-Denis, Val-de-Marne) — partly to see if the idea was tractable, and partly to surface the methodological problems early rather than at thesis stage.

This repo is that pipeline. It has two independent but complementary parts:

| # | Notebook | Question it asks |
|---|----------|-------------------|
| 1 | [`01_trajectory_typology_clustering.ipynb`](notebooks/01_trajectory_typology_clustering.ipynb) | Can commune-level job growth (SIRENE) and housing-market dynamics (DVF) be combined into a data-driven **typology of territorial trajectories** — the kind of "which areas are gaining jobs but losing affordability" pattern LUTI vulnerability analysis cares about? |
| 2 | [`02_business_birth_forecast_xgboost.ipynb`](notebooks/02_business_birth_forecast_xgboost.ipynb) | Can a two-stage model (linear trend + gradient-boosted residual correction) **forecast next-year business creation** better than a naive persistence baseline? |

## Key findings

![Trajectory clusters by commune](figures/map_typology_clusters.png)

- **Typology (notebook 1):** clustering a 4-feature commune panel (job-base growth, price growth, transaction-volume change, price level) with k-means recovers a four-way typology that maps cleanly onto known geography — central Paris (low job growth, high price/activity), affluent inner-west suburbs (stable, low growth), most of Seine-Saint-Denis (high job growth, still affordable), and a smaller "gentrifying" pocket on the northeastern edge combining job growth, price growth *and* activity from a low starting price level. It also surfaced a real weakness first: one or two extreme communes visibly dominate the price-growth-rate feature, which is exactly the outlier risk flagged in §4 of the proposal and motivates the winsorization/stability checks planned for the full thesis.
- **Forecasting (notebook 2):** benchmarked against a naive persistence baseline on held-out 2025 business-birth data. The result is a genuine trade-off, not a clean win either way — **persistence has the lowest typical-case error (MAE) but by far the worst error on extreme communes (RMSE/R²)**, while the ML models invert that pattern. Adding more training history (2016–2021 vs. 2016-only) improved MAE but made RMSE/R² worse, and the gradient-boosted residual correction added almost nothing over the linear component alone. BPE amenity features were tested and dropped — near-zero standalone predictive power once business history is already in the model.

Full diagnostics, tables, and the feature-level choropleths behind the typology are in [`docs/results.md`](docs/results.md). Both notebooks were run locally against processed parquet panels; see [Data](#data--reproducing) below for what's needed to rerun them.

## Repo structure

```
grand-paris-repo/
├── README.md
├── LICENSE
├── requirements.txt
├── docs/
│   ├── proposal_context.md      # how each notebook maps to the HARMONIC proposal's objectives
│   └── results.md               # full diagnostics: cluster maps explained, XGBoost tables, error analysis
├── figures/
│   ├── map_typology_clusters.png
│   └── map_typology_features.png
└── notebooks/
    ├── 01_trajectory_typology_clustering.ipynb
    └── 02_business_birth_forecast_xgboost.ipynb
```

## Data & reproducing

The notebooks run on open French administrative data, not redistributed here (large, and some require registration):

- **SIRENE** — establishment creation/closure records (INSEE) → [sirene.fr](https://www.sirene.fr/) / [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/)
- **DVF** ("Demandes de Valeurs Foncières") — property transaction records (DGFiP) → [app.dvf.etalab.gouv.fr](https://app.dvf.etalab.gouv.fr/)
- **BPE** ("Base Permanente des Équipements") — facility/amenity inventory (INSEE) → [insee.fr](https://www.insee.fr/fr/statistiques/3568638)
- **SIDE** — business creation/death statistics by commune (INSEE)

The notebooks expect these already downloaded and pre-processed into commune-level parquet panels under a local `data/raw/` and `data/processed/` directory — the `BASE`/`PATH` variables at the top of each notebook point to where I keep them locally and will need updating to match your own layout. Nothing in this repo depends on a proprietary or paid source.

```bash
git clone https://github.com/<your-username>/grand-paris-territorial-typology.git
cd grand-paris-territorial-typology
python -m venv .venv && source .venv/bin/activate   # or your preferred env manager
pip install -r requirements.txt
jupyter lab
```

Then update the `BASE`/`PATH` constants near the top of each notebook to point at your local data, and run top to bottom.

## Stack

`pandas` · `numpy` · `scikit-learn` (KMeans, StandardScaler, silhouette score) · `xgboost` · `geopandas` + `contextily` (spatial validation / choropleths) · `matplotlib`

## Author

**Ahmed Elbukhari** — Research Assistant, Transport and Traffic Modelling, Linköping University (CODE FLOW project). MSc Intelligent Transportation Systems & Logistics (Linköping); BSc Civil Engineering–Transportation (University of Khartoum). Co-lead, KhartouMap (MIT Open Data Prize, 2024).

<!-- Add your LinkedIn / Scholar / contact links here -->

## License

MIT — see [LICENSE](LICENSE). Code only; the underlying open datasets carry their own licenses (Etalab Open Licence for DVF/SIRENE/BPE).
