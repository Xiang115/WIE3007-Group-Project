# WIE3007-Group-Project

## Run feature engineering and logistic regression

Install the notebook dependencies in the same Python environment you select as the Jupyter kernel. Use `python -m pip install -r requirements.txt`. On Windows, you can request the prebuilt CPU wheel with:

```powershell
python -m pip install --only-binary=llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
```

1. Run [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb) to build `data/loan_features.csv`, `data/train.csv`, and `data/test.csv` from `data/financial_loan_dataset_v2.csv`. The first run downloads the 812 MB [Qwen3.5-0.8B text checkpoint](https://ollama.com/library/qwen3.5:0.8b) into `.cache/`. Inference runs locally through llama.cpp on CPU or a supported GPU. Thinking mode generates up to 32 reasoning tokens per remark and prompt, so the first run can take substantially longer. Later runs reuse scores for each mode when the remarks and prompt settings match.
2. Run [04b_model_logistic_regression.ipynb](notebooks/04b_model_logistic_regression.ipynb) to compare a tabular-only baseline with Qwen3.5 non-thinking and thinking features on the same test split and write `results/logistic_regression_metrics.csv`.

Use the same `train.csv` and `test.csv` for each group model. The target is `loan_default` (1 means default). The feature tables retain `customer_id` for joining results and `customer_remark` for reference. Exclude both from the predictor matrix. Fit model-specific preprocessing on training data.

The exported `slm_` columns and `customer_segment` come from Qwen3.5. The numeric feature lists in notebook 03 use the non-thinking sentiment and strain scores by default.
