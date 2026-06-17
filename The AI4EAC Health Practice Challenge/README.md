# Predicting Depression from Routine Household Survey Data

A machine learning project predicting depression risk among individuals in rural Kenya using routine household economic and demographic survey data, based on the [Zindi AI4EAC Health Practice Challenge](https://zindi.africa/competitions/the-ai4eac-health-practice-challenge) hosted by the Busara Center for Behavioral Economics.

## Project Overview

The World Health Organization estimates that 1.3 million Kenyans suffer from untreated major depressive disorder each year, with sub-Saharan Africa recording the highest regional prevalence in the world. With only two certified psychiatrists per million people in Kenya, most communities — particularly rural areas — have little to no access to mental health services.

This project explores whether **routine survey data already collected for other purposes** (economic activity, food security, household assets, health expenditure, education) can help identify individuals who may be experiencing depression, with the goal of supporting **smart targeting** by community health workers, NGOs, or local clinics with limited resources.

## Data

- **Source**: Busara Center for Behavioral Economics, via Zindi (AI4EAC Health Practice Challenge)
- **Observations**: 1,143 individuals (training data)
- **Target variable**: `depressed` — whether the respondent meets the epidemiological threshold for moderate depression
- **Class distribution**: 16.9% depressed (193), 83.1% not depressed (950) — substantially imbalanced
- **Features**: ~75 variables spanning demographics, household assets, consumption/expenditure, income and enterprise activity, food security, medical/health indicators, education, and mobile money (M-Pesa) usage

## Methodology

### 1. Exploratory Data Analysis

Variables were grouped into conceptual domains (demographics, assets, consumption, income/enterprise, food security, medical/health, education, M-Pesa) and compared between depressed and non-depressed groups using bivariate statistical tests.

Across all domains, only a handful of variables showed statistically significant univariate associations with depression status (p < 0.05): `asset_durable`, `nondurable_investment`, `ent_nonag_flowcost`, `ent_total_cost`, `fs_adwholed_often` (adult food deprivation), and `med_portion_sickinjured` (proportion of household sick/injured). The consistent direction of these findings — lower asset/investment levels and greater material/health hardship among depressed respondents — motivated a multivariate, machine-learning-based approach capable of detecting feature interactions that bivariate tests cannot capture.

### 2. Preprocessing

A systematic preprocessing pipeline was applied and tracked at each stage:

| Pre-processing step | Variables removed |
|---|---|
| Survey identifiers (id, village, date) | 3 |
| Constant variables | 3 |
| >30% missing | 16 |
| Redundant derived variable (age_group) | 1 |
| Above 0.8 correlated features | — |
| **Final feature count** | **~50** |

Remaining missing data (8 variables, <30% missing each) was imputed using **Multiple Imputation by Chained Equations (MICE)**, with post-imputation constraints applied based on each variable's definition (e.g., binary indicators clipped to {0,1}, sick days capped at 30/month).

### 3. Handling Class Imbalance

Given the substantial class imbalance (16.9% positive class), **SMOTE-ENN** (Synthetic Minority Over-sampling + Edited Nearest Neighbours) was applied to the training data. To avoid data leakage, resampling, scaling, and model fitting were combined into a single `imblearn` pipeline applied **within each cross-validation fold**, ensuring the validation set always reflected the true, imbalanced class distribution.

### 4. Model Selection

Seven classifiers were evaluated — Logistic Regression, Elastic Net, Random Forest, SVM, KNN, XGBoost, and LightGBM — each tuned via a randomized search over 8 hyperparameter configurations with 5-fold cross-validation (F1 scoring), then evaluated on a held-out validation set (20% of data, original class distribution preserved).

**XGBoost** achieved the best validation performance and was selected as the final model.

### 5. Final Model Performance (Validation Set, n=229)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Not Depressed (0) | 0.88 | 0.71 | 0.78 | 190 |
| Depressed (1) | 0.26 | 0.51 | 0.35 | 39 |
| **Accuracy** | | | **0.67** | 229 |
| Macro avg | 0.57 | 0.61 | 0.56 | 229 |
| Weighted avg | 0.77 | 0.67 | 0.71 | 229 |

The model identifies just over half (51%) of true depression cases (recall), at the cost of a high false-positive rate (precision of 0.26 for the depressed class). In a screening context — where missing a true case is more costly than a false alarm that triggers further (low-cost) follow-up — this recall/precision tradeoff may be acceptable, though it should be weighed carefully against the resource cost of follow-up.

### 6. Interpretability (SHAP)

SHAP (SHapley Additive exPlanations) values were used to identify the 15 most influential features for the final model, including `med_sickdays_hhave`, `fs_adwholed_often` (adult food deprivation), `married`, `asset_savings`, `asset_livestock`, `age`, `fs_sleephun`, `ent_farmexpenses`, `med_portion_sickinjured`, `ent_ownfarm`, `asset_durable`, `fs_enoughtom`, `cons_ed`, and `edu`. These broadly reflect the same domains (food security, household economic activity, health burden) flagged during bivariate EDA.

## Interactive Application

The final model (top 15 SHAP features) is deployed as a **Streamlit app**, allowing a user to input survey responses and receive:

- An estimated probability of depression
- A risk tier (Low / Medium / High)
- An individualized SHAP-based explanation of which factors drove the prediction

> ⚠️ **This tool is for educational/portfolio purposes only.** It is not a diagnostic instrument and should not be used to make clinical decisions.

### Running the app locally

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Repository Structure

```
.
├── data/           # Raw and processed data (not included - see Data section)
├── notebooks/       # EDA, preprocessing, and modeling notebooks
├── src/             # Helper functions (preprocessing tracker, etc.)
├── models/          # Saved model bundle (.pkl)
├── app/             # Streamlit application
├── requirements.txt
└── README.md
```

## Limitations & Future Work

- **Modest predictive performance**: validation F1 for the depressed class (~0.35) reflects the inherent difficulty of detecting depression from indirect economic/behavioral proxies. This is consistent with the weak bivariate associations found during EDA.
- **Small sample size** (n=1,143) limits the feasibility of deep learning approaches, which typically require substantially more data.
- **Future directions**: incorporating direct psychosocial indicators (social support, recent life events, sleep quality), exploring deep learning with larger/pooled datasets, threshold optimization based on the precision-recall tradeoff for the intended use case, and longitudinal data capturing changes in circumstances over time.

A more detailed technical report, including full EDA tables and methodology, is available in [`reports/technical_report.pdf`](reports/technical_report.pdf).

## Recommendations for Practitioners

For organizations (NGOs, clinics, community health programs) considering a similar approach, the following recommendations should be considered before deployment:

- **Use as a triage aid, not a diagnostic tool.** Given the recall/precision tradeoff observed (51% recall at 26% precision for the depressed class), this type of model is best positioned as a low-cost first-pass filter to help prioritize which households a community health worker visits first — not as a substitute for clinical screening (e.g., PHQ-9 or similar validated instruments) administered by trained personnel.

- **Pair the model with a human-in-the-loop process.** Flagged individuals should be referred for a brief follow-up conversation or validated screening tool, rather than acted on directly. This mitigates the cost of false positives (unnecessary outreach) while still capturing the benefit of improved targeting relative to random or unguided outreach.

- **Budget for the false-positive rate.** With ~26% precision, roughly 3 in 4 individuals flagged as "high risk" will not meet the depression threshold. Programs should size follow-up capacity accordingly, and communicate this expectation to field staff to avoid eroding trust in the tool.

- **Re-validate before scaling to new regions or populations.** The model was trained on survey data from a specific set of villages/communities. Economic indicators, food security patterns, and their relationship to depression may vary across regions (urban vs. rural, different agro-ecological zones, different economic baselines). Local re-validation — ideally with a small labeled sample from the new context — is recommended before relying on the model's outputs operationally.

- **Monitor for data drift and fairness.** Periodically re-check model performance as economic conditions change (e.g., drought, inflation, policy shifts affecting smallholder farmers) since the model's key drivers — food security, farm income, asset values — are themselves sensitive to such shocks. Performance should also be checked across subgroups (e.g., by gender, marital status, region) to ensure the model does not systematically under- or over-flag particular populations.

- **Treat this as a complement to, not a replacement for, investment in mental health infrastructure.** Improved targeting only delivers value if there is capacity to act on it — referral pathways, trained personnel, and follow-up resources are prerequisites for this type of tool to translate into improved outcomes.

## Acknowledgements

Data provided by the **Busara Center for Behavioral Economics**, in partnership with AI Kenya, Women in Machine Learning and Data Science, Tulaa, and ALX Launchpad, via [Zindi](https://zindi.africa/competitions/the-ai4eac-health-practice-challenge).
