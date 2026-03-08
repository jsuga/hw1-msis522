# NYC Airbnb End-to-End Data Science Pipeline

This project delivers a full pipeline for the NYC Airbnb Open Data dataset (`AB_NYC_2019.csv`):
- exploratory data analysis (EDA)
- cross-validated model benchmarking
- holdout evaluation
- SHAP explainability for every trained model
- Streamlit dashboard with an executive summary and interactive feature playground

## Models
The training pipeline includes:
- Linear Regression
- Lasso Regression
- Ridge Regression
- CART (Decision Tree Regressor)
- Random Forest Regressor
- LightGBM Regressor (if `lightgbm` is installed)

## Data
Place the dataset at the project parent level (one directory above `archive/`):
`../AB_NYC_2019.csv`

## Run
From the `archive/` folder:

```bash
pip install -r requirements.txt
python train_and_save_models.py
streamlit run app.py
```

Optional SHAP artifacts + plots:

```bash
python build_artifacts.py
```

## Generated Artifacts
Saved under `archive/artifacts/`:
- `models/*.joblib`: fitted model pipelines
- `metadata/*.json`: app-ready metrics, model manifest, predictions, feature defaults, and summary data
- `metrics.json`: CV means/std/fold scores + holdout metrics (from `build_artifacts.py`)
- `summary.json`: run metadata and best model (from `build_artifacts.py`)
- `diagnostics.json`: residual diagnostics payload (from `build_artifacts.py`)
- `shap_index.json`: pointers to SHAP files per model (from `build_artifacts.py`)
- `shap/<model_name>/...`: SHAP arrays, transformed data, and importance table
- `plots/shap_*.png`: summary and bar SHAP plots
