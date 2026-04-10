# Cleaning Summary
**Dataset:** `data/raw/data.csv` | **random_state:** 42 | **Date:** 2026-04-10

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Raw shape | 6,819 rows × 96 columns |
| Feature dtypes | 93 float64, 3 int64 |
| Missing values | **None** |
| Target column | `Bankrupt?` |
| Bankrupt (1) | 220 rows (3.23%) |
| Solvent (0) | 6,599 rows (96.77%) |

---

## 2. Column Cleaning

| Decision | Detail |
|---|---|
| Strip column spaces | All 96 names had a leading space (e.g., `" Bankrupt?"`) — stripped with `str.strip()` as first operation |
| Drop zero-variance | `Net Income Flag` — constant across all rows; carries no information and causes divide-by-zero in scalers |
| Flag binary columns | `Liability-Assets Flag` — 2 unique values; saved to `data/processed/binary_columns.txt`; not scaled |

After cleaning: **6,819 rows × 95 columns** (94 features + 1 target).

---

## 3. Train / Val / Test Split

**Method:** Two-stage stratified split, `random_state=42`.

| Split | Rows | Bankrupt | Solvent | Bankrupt % |
|---|---|---|---|---|
| Train | 4,773 | 154 | 4,619 | 3.23% |
| Val | 1,023 | 33 | 990 | 3.23% |
| Test | 1,023 | 33 | 990 | 3.23% |

Stratification confirmed exact across all three splits.
Raw unscaled splits saved to `data/processed/train.csv / val.csv / test.csv`.

---

## 4. Pipeline Versions (A–E)

All scalers and transformers are **fit on train only**. Val and test are only transformed (never fit). SMOTE is only applied to train.

| Version | Scaler | Extra | Train rows | Bankrupt % | Features | Source |
|---|---|---|---|---|---|---|
| A | StandardScaler | — | 4,773 | 3.23% | 94 | Our pipeline |
| B | RobustScaler | SMOTE | 9,238 | 50.00% | 94 | Our pipeline |
| C | StandardScaler | Corr pruning >0.95 | 4,773 | 3.23% | 74 | Our pipeline |
| D | StandardScaler | PCA 95% variance | 4,773 | 3.23% | 50 PCs | Our pipeline |
| E | StandardScaler | Feature engineering + selection | 4,773 | 3.23% | 16 | Friend's v2, reproduced on our split |

### Version A — StandardScaler, no resampling
- 94 original features, zero-mean unit-variance
- Natural 3.23% imbalance kept — handle with `class_weight='balanced'` at training time
- **For:** Logistic Regression, Linear SVM

### Version B — RobustScaler + SMOTE
- RobustScaler (median/IQR) reduces influence of extreme outliers (max skewness ~82)
- SMOTE (k=5) applied only to train → 4,773 → 9,238 rows, 50/50 balanced
- Val and test NOT resampled — they keep the natural distribution for realistic evaluation
- **For:** XGBoost, Random Forest, LightGBM

### Version C — StandardScaler + Correlation Pruning
- Dropped 20 features with pairwise |r| > 0.95 (computed on train only)
- 94 → 74 features; dropped column list: `data/processed/version_C_dropped_columns.txt`
- **For:** MLP — removes redundant inputs that slow convergence

### Version D — StandardScaler + PCA
- PCA fit on train, retaining 95.46% variance in 50 components
- Column names: `PC1` … `PC50`; metadata: `outputs/pca_info.json`
- Removes multicollinearity completely
- **For:** MLP, TabNet

### Version E — Feature Engineering + Aggressive Selection (Friend's v2)
Pipeline applied on **our split** (same rows as A–D):
1. **Feature engineering** — 12 new features added (growth, ratios, interactions):
   - Growth: `feat_roa_momentum`, `feat_leverage_growth`, `feat_liquidity_trend`
   - Ratios: `feat_debt_service_ratio`, `feat_prof_to_debt`, `feat_asset_quality`, `feat_quick_to_current`, `feat_cfo_to_operating`
   - Interactions: `feat_profit_x_leverage`, `feat_liquidity_x_debt`, `feat_cashflow_x_liability`, `feat_financial_volatility`
2. **Imputation** — median (fit on train; no nulls in raw data but inf/NaN can appear from divisions)
3. **Outlier clipping** — 1st/99th percentile (fit on train)
4. **Correlation pruning** — drop |r| > 0.95 → 81 features
5. **Low variance filter** — drop var < 0.01 → 20 features
6. **Target correlation filter** — keep |corr with Bankrupt?| > 0.01 → **16 features**
7. **StandardScaler** (fit on train)
- Selected features saved to `data/processed/version_E_selected_columns.txt`
- **For:** Logistic Regression, interpretable/explainable ML where feature count matters

Top features by target correlation:
| Feature | |r with target| |
|---|---|
| `feat_financial_volatility` | 0.3145 |
| `feat_asset_quality` | 0.1396 |
| `Tax rate (A)` | 0.1179 |
| `Cash/Total Assets` | 0.1007 |
| `feat_quick_to_current` | 0.0874 |

---

## 5. Top-10 Most Skewed Features (raw)

Max skewness ~82 confirms extreme outliers in financial ratios — justifies RobustScaler for Version B and outlier clipping in Version E.

| Feature | Skewness |
|---|---|
| Fixed Assets to Assets | 82.58 |
| Current Ratio | 82.58 |
| Total income/Total expense | 82.33 |
| Net Value Growth Rate | 80.29 |
| Contingent liabilities/Net worth | 79.67 |

---

## 6. Top Correlated Pairs (raw)

Multiple perfectly correlated pairs (|r|=1.0) confirmed — justifies the >0.95 pruning in versions C and E.

| Col A | Col B | r |
|---|---|---|
| Current Liabilities/Liability | Current Liability to Liability | 1.000 |
| Debt ratio % | Net worth/Assets | −1.000 |
| Operating Gross Margin | Gross Profit to Sales | 1.000 |

---

## 7. Combining Our Pipeline and Friend's Pipeline

| Aspect | Our pipeline (A–D) | Friend's pipeline (v2 → E) |
|---|---|---|
| Split method | 70% train, then 30% temp → 50/50 val/test | 85% trainval → 15% test, then ~82/18 train/val |
| Split rows | Same final counts (4773/1023/1023) | Same final counts but **different row assignments** |
| Feature engineering | None (raw financial ratios) | 12 engineered features added |
| Outlier handling | RobustScaler (Version B) | 1-99% clipping before any step |
| Dimensionality reduction | Corr pruning (C) or PCA (D) | Corr + low-var + target-corr selection |
| Final features | 94 / 74 / 50 | **16** |

**Resolution:** Version E was reproduced by re-running the friend's pipeline steps on **our** split, so all five versions share identical row assignments. This is required for fair model comparison.

---

## 8. What Was NOT Done (and Why)

| What | Why |
|---|---|
| Imputation | No missing values in raw data |
| Log/power transform | Deferred to modelling; tree models don't need it; DL uses PCA instead |
| SMOTE on val/test | Would corrupt evaluation — natural imbalance preserved |
| Leakage | Enforced: all fitting done on train only |
| Feature selection for A/B | Kept all 94 to give each model full signal |

---

*random_state=42 throughout all steps.*
