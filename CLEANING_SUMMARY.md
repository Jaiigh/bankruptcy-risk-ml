# Cleaning Summary

## 1. Dataset Overview

| Property | Value |
|---|---|
| Raw shape | 6,819 rows × 96 columns |
| Feature dtypes | 93 float64, 3 int64 |
| Missing values | **None** (0 nulls across all columns) |
| Target column | `Bankrupt?` |
| Bankrupt count | 220 (3.23%) |
| Solvent count | 6,599 (96.77%) |

The dataset is heavily imbalanced: only 1 in ~31 companies is bankrupt. This drives all resampling decisions downstream.

---

## 2. Column Cleaning Decisions

### 2a. Leading-space stripping
All 96 column names in `data/raw/data.csv` carry a **leading space** (e.g., `" Bankrupt?"` → `"Bankrupt?"`). Every column name was stripped with `df.columns.str.strip()` as the very first operation before any analysis or splitting.

### 2b. Zero-variance column dropped
| Column | Reason |
|---|---|
| `Net Income Flag` | `nunique() == 1` — constant across all 6,819 rows; carries zero information and would cause divide-by-zero in scalers |

After dropping: **6,819 rows × 95 columns** (94 features + 1 target).

### 2c. Binary columns flagged
One binary (flag-type) feature was identified:

| Column | Unique values |
|---|---|
| `Liability-Assets Flag` | 2 |

Saved to `data/processed/binary_columns.txt`. These columns may benefit from special treatment (no scaling needed; logistic regression can use them as-is).

---

## 3. Train / Val / Test Split

**Rationale:** Stratified split is mandatory because the 3.23% minority class is so small that a random split would produce highly variable class proportions across folds.

**Method:** Two-stage stratified split using `sklearn.model_selection.train_test_split` with `stratify=y, random_state=42`.

| Split | Rows | Bankrupt | Solvent | Bankrupt % |
|---|---|---|---|---|
| Train | 4,773 | 154 | 4,619 | 3.23% |
| Val | 1,023 | 33 | 990 | 3.23% |
| Test | 1,023 | 33 | 990 | 3.23% |
| **Total** | **6,819** | **220** | **6,599** | **3.23%** |

Stratification is confirmed exact — all three splits maintain the same 3.23% minority rate.

**Raw cleaned CSVs saved to:**
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

---

## 4. Pipeline Versions

All scalers are **fit on train only** and applied to val/test — no leakage. SMOTE is applied **after scaling**, **on train only** — val and test are never resampled.

| Version | Scaler | Resampling | Train rows | Train bankrupt % | Val rows | Test rows |
|---|---|---|---|---|---|---|
| A | StandardScaler | None | 4,773 | 3.23% | 1,023 | 1,023 |
| B | StandardScaler | SMOTE (k=5) | 9,238 | 50.00% | 1,023 | 1,023 |
| C | RobustScaler | None | 4,773 | 3.23% | 1,023 | 1,023 |
| D | RobustScaler | SMOTE (k=5) | 9,238 | 50.00% | 1,023 | 1,023 |

SMOTE synthetically generates minority-class samples until train reaches 50/50 balance (154 → 4,619 bankrupt samples added; total train 4,773 → 9,238).

**Version CSVs saved to `outputs/`** (e.g. `version_A_train.csv`, `version_A_val.csv`, etc.)
**Summary table:** `outputs/version_summary.csv`

---

## 5. Recommended Version per Model

| Model | Recommended Version | Reason |
|---|---|---|
| Logistic Regression | A | Assumes normally-distributed features; handle imbalance via `class_weight='balanced'` rather than synthetic data |
| Linear SVM | A or C | Linear SVMs are robust to scale but sensitive to outliers; use C if many outliers are present |
| RBF-SVM / KNN | C | Distance-based models need RobustScaler to reduce outlier distortion; no SMOTE needed with `class_weight` |
| RBF-SVM / KNN (max recall) | D | Same as C but SMOTE-balanced training maximises minority-class recall at the cost of more false positives |
| Random Forest / XGBoost | A or B | Tree models are scale-invariant but benefit from explicit resampling when minority class < 5%; B gives balanced signal |
| LightGBM / CatBoost | A | Native `scale_pos_weight` parameter handles imbalance; synthetic data not needed |
| MLP / Neural Network | B | Neural nets train more stably on balanced batches; StandardScaler suits gradient-based optimisation |

---

## 6. Top-10 Most Skewed Features

Extreme skewness (>60) confirms that raw features have long tails — supporting the use of RobustScaler for outlier-sensitive models.

| Rank | Feature | Skewness |
|---|---|---|
| 1 | Fixed Assets to Assets | 82.58 |
| 2 | Current Ratio | 82.58 |
| 3 | Total income/Total expense | 82.33 |
| 4 | Net Value Growth Rate | 80.29 |
| 5 | Contingent liabilities/Net worth | 79.67 |
| 6 | Realized Sales Gross Profit Growth Rate | 77.93 |
| 7 | Operating Profit Growth Rate | −71.69 |
| 8 | Operating Profit Rate | −70.24 |
| 9 | Continuous Net Profit Growth Rate | 67.10 |
| 10 | Total Asset Return Growth Rate Ratio | 62.50 |

---

## 7. Top-10 Most Correlated Feature Pairs

Several feature pairs are **perfectly correlated (|r| = 1.0)**, indicating redundant columns. Feature selection / dimensionality reduction (PCA, VIF pruning) is recommended in the modelling phase.

| Col A | Col B | Correlation |
|---|---|---|
| Current Liabilities/Liability | Current Liability to Liability | 1.000 |
| Current Liabilities/Equity | Current Liability to Equity | 1.000 |
| Debt ratio % | Net worth/Assets | −1.000 |
| Operating Gross Margin | Gross Profit to Sales | 1.000 |
| Net Value Per Share (A) | Net Value Per Share (C) | 0.9998 |
| Operating Gross Margin | Realized Sales Gross Margin | 0.9995 |
| Realized Sales Gross Margin | Gross Profit to Sales | 0.9995 |
| Net Value Per Share (B) | Net Value Per Share (A) | 0.9993 |
| Net Value Per Share (B) | Net Value Per Share (C) | 0.9992 |
| Operating Profit Per Share (Yuan ¥) | Operating profit/Paid-in capital | 0.9987 |

---

## 8. What Was NOT Done (and Why)

| What | Why not done |
|---|---|
| Imputation | No missing values — not needed |
| Outlier removal | Retained to let models handle naturally; RobustScaler mitigates impact |
| Feature selection / VIF pruning | Deferred to modelling phase — kept all 94 features to give each model full signal |
| Log/power transformation | Deferred; tree models don't need it and linear models use StandardScaler |
| Class-weight tuning | Model hyperparameter — set per-model during training, not in preprocessing |
| Data leakage guard | **Enforced**: all scalers/SMOTE fit on train only; val and test never seen during fitting |

---

*Generated: 2026-04-10 | random_state=42 throughout*
