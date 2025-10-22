# Cash Advance Default Modelling

This repository contains code, data, and artifacts used to build and evaluate a model that predicts defaults on cash advance products. It contains a small local modelling library, a Jupyter notebook used for exploratory data analysis and model training, sample data and pre-computed artifacts (features, dropped features, model pickle, and PDP plots).

## Project structure

- `data/`
  - `Data.csv` — primary dataset used for modelling.
  - `Data_dictionary.csv` — data dictionary describing columns in `Data.csv`.

- `Local_Model/` — lightweight modelling helpers used by the notebook and experiments.
  - `base_model.py` — base model implementation (cross-validated training, evaluation helpers).
  - `base_model_without_cv.py` — variant of base model without cross-validation.
  - `imputation.py` — imputation utilities used to fill missing values prior to model training.
  - `EarlyStoppingMaximization.py`, `EarlyStoppingMinimization.py` — early stopping callbacks/utilities used during hyperparameter search or training.

- `Artifacts/` — generated outputs from model runs and feature engineering.
  - `model.pickle` — trained and serialized model object (pickle). Use with caution; pickle files can execute code when deserialized. Prefer re-training in a fresh environment when possible.
  - `features.json`, `Features_0.90_pctNULL.json`, `Features_0.99_pctNULL.json` — feature lists / selection results from pre-processing steps.
  - `dropped_features.json` — features dropped during pre-processing and reasoning.
  - `optuna_study_local.db` — Optuna study DB produced by hyperparameter tuning runs.
  - `PDP_PLOTS.pdf` — partial dependence plots and feature effect visualizations.

- `Notebook/`
  - `Modelling_Cash_Advances.ipynb` — main exploratory notebook that demonstrates data loading, cleaning, feature selection, model training, hyperparameter tuning, and evaluation.

- project root
  - `utils.py` — small utility helpers used by the notebook or scripts.
  - `README.md` — (this file)
  - `LICENSE` — licensing terms for the repository.


## Goals and scope

The goals of this repository are:

- Demonstrate a reproducible workflow for modelling cash advance default risk using the provided dataset.
- Provide modular code (in `Local_Model/`) for reusable model training, imputation and early stopping utilities.
- Store experiment artifacts (selected features, PDPs, model pickle, and hyperparameter study database) to support analysis and reporting.

This repository is oriented towards exploration and reproducible experiments rather than a production-ready deployment.



## Quickstart (recommended)

This repository does not include a pinned `requirements.txt`. The notebook contains a watermark cell that records the active environment and installed package versions used during development. Use that watermark to reproduce the same environment precisely.

1. Create and activate a Conda environment (recommended using Python 3.9+). Example using conda:

  - Create and activate (replace `cad-env` with your preferred name):

    ```powershell
    conda create -n cad-env python=3.9 -y
    conda activate cad-env
    python -m pip install --upgrade pip
    ```

2. Open `Notebook/Modelling_Cash_Advances.ipynb` in Jupyter Notebook or VS Code and locate the watermark cell near the top of the notebook. The watermark lists the exact package versions (for example, pandas, numpy, scikit-learn, optuna, matplotlib, seaborn, joblib) used when the notebook was last run in this environment.

3. Install the packages and versions shown in the watermark into your activated virtual environment. Example (replace versions with what the watermark shows):

  ```powershell
  pip install pandas==1.5.3 numpy==1.24.2 scikit-learn==1.2.2 optuna==3.2.0 matplotlib==3.7.1 seaborn==0.12.2 joblib==1.2.0 notebook
  ```

4. Run the notebook cells in order. The notebook loads `data/Data.csv`, uses utilities from `Local_Model/` to preprocess and train the model, and saves artifacts to `Artifacts/` when requested.


## Reproducing training from the notebook

The notebook contains the step-by-step code used to preprocess the data, run feature selection, tune hyperparameters via Optuna, and train the final model. Typical steps are:

1. Load `data/Data.csv` into a pandas DataFrame.
2. Inspect and apply imputation rules in `Local_Model/imputation.py`.
3. Select or engineer features; consult `Artifacts/features.json` and `Artifacts/dropped_features.json` for previously used lists.
4. (Optional) Run Optuna hyperparameter tuning — the notebook will write to `Artifacts/optuna_study_local.db`.
5. Train the final model using `Local_Model/base_model.py` or `Local_Model/base_model_without_cv.py`.
6. Save the trained model to `Artifacts/model.pickle` and generate diagnostic plots (PDPs, calibration, ROC).

Notes:
- The existing `model.pickle` is a convenience artifact. Because pickle can embed environment-specific state, re-training in your environment is recommended before trusting the model for new predictions.


## Using the trained model

To load and use the provided `Artifacts/model.pickle` for inference in a trusted environment:

```python
import pickle
from pathlib import Path

model_path = Path('Artifacts') / 'model.pickle'
with model_path.open('rb') as f:
    model = pickle.load(f)

# Prepare a pandas DataFrame X with the same columns used for training, then:
# preds = model.predict_proba(X)[:, 1]  # if model implements predict_proba
```

Security note: only load pickle files from trusted sources.


## Artifacts and how to interpret them

- `model.pickle`: serialized model object — may be an sklearn pipeline or custom wrapper. Check the notebook to see the exact object type.
- `optuna_study_local.db`: contains the Optuna trials and parameter search results. Optuna can be reloaded from this file to inspect the search and reproduce the best trials.
- `PDP_PLOTS.pdf`: Partial dependence plots for top features — use as diagnostic to understand marginal feature effects.
- `features.json`, `Features_0.90_pctNULL.json`, `Features_0.99_pctNULL.json`, `dropped_features.json`: these files capture feature selection decisions. They are useful when constructing the production feature set.


## Development notes and code layout

- `Local_Model/base_model.py` — likely contains training loops, cross-validation wrappers, metric calculations and model saving utilities. Read the file to adapt behavior or integrate with other pipelines.
- `Local_Model/imputation.py` — contains strategies used to fill missing values before model training. Align inference-time imputation with training-time imputation.
- Early stopping utilities — used to manage hyperparameter tuning or iterative training.

If you plan to extend or refactor the code:

- Add a `requirements.txt` or `environment.yml` to pin dependencies.
- Add small unit tests for `Local_Model` utilities and for the main preprocessing functions.
- Add a script like `train.py` to reproduce the notebook without manual cell execution.


## Reproducibility checklist

- Pin package versions in `requirements.txt`.
- Use a deterministic random seed when training and tuning.
- When using Optuna, consider saving the study database and recording the best trial parameters in a JSON or YAML file.


## Limitations and cautions

- The dataset and model artifacts in this repo are for experimental and interview/demo purposes. They have not been hardened for production deployment.
- The serialized model (`model.pickle`) may not be portable across Python versions or environments.
- PII and sensitive fields: ensure the dataset has been cleared of personal identifiable information before sharing or deploying.


## License

See the `LICENSE` file in the repository root for license terms.
# Cash_Advance_Default_Modelling