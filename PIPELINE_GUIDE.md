# Pipeline Guide — How to Use the Processed Data

## Version quick-reference

| Version | File prefix | Scaler | Extra | Features | Use for |
|---------|------------|--------|-------|---------|---------|
| A | `outputs/version_A_*.csv` | StandardScaler | — | 94 | Logistic Regression, Linear SVM |
| B | `outputs/version_B_*.csv` | RobustScaler | SMOTE | 94 | XGBoost, Random Forest, LightGBM |
| C | `outputs/version_C_*.csv` | StandardScaler | Corr pruning | 74 | MLP (feature-reduced) |
| D | `outputs/version_D_*.csv` | StandardScaler + PCA | — | 50 PCs | MLP, TabNet |
| E | `outputs/version_E_*.csv` | StandardScaler | Feature eng + selection | 16 | Logistic Reg, explainable ML |

---

## Standard loading pattern (same for every version)

```python
import pandas as pd

train = pd.read_csv("outputs/version_X_train.csv")   # replace X with A/B/C/D/E
val   = pd.read_csv("outputs/version_X_val.csv")
test  = pd.read_csv("outputs/version_X_test.csv")

X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]
X_val,   y_val   = val.drop(columns=["Bankrupt?"]),   val["Bankrupt?"]
X_test,  y_test  = test.drop(columns=["Bankrupt?"]),  test["Bankrupt?"]
```

---

## Role 2 — Baseline ML

### Version A → Logistic Regression, Linear SVM
```python
from sklearn.linear_model import LogisticRegression

train = pd.read_csv("outputs/version_A_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

model = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
model.fit(X_train, y_train)
```
> Use `class_weight='balanced'` — Version A keeps the natural 3.23% imbalance.

### Version B → XGBoost, Random Forest, LightGBM
```python
from xgboost import XGBClassifier

train = pd.read_csv("outputs/version_B_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
```
> No `class_weight` needed — SMOTE already balanced train to 50/50.

---

## Role 3 — Deep Learning

### Version C → MLP with named features (74 cols)
```python
train = pd.read_csv("outputs/version_C_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]
# Correlated redundancy removed — cleaner signal for gradient descent
```

### Version D → MLP / TabNet with PCA input (50 cols: PC1…PC50)
```python
train = pd.read_csv("outputs/version_D_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]
# Columns are PC1…PC50 — fully uncorrelated, 95.46% variance retained
```

---

## Bonus: Version E — Interpretable / Feature-Engineered (16 cols)

Reproduced from friend's v2 pipeline on our split. 16 hand-crafted + selected features with the highest correlation to bankruptcy. Good for:
- Logistic Regression baseline (interpretable coefficients)
- Comparing "domain knowledge" features vs raw features
- SHAP analysis (fewer features → cleaner explanations)

```python
train = pd.read_csv("outputs/version_E_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

# Top features include: feat_financial_volatility, feat_asset_quality,
# Tax rate (A), Cash/Total Assets, feat_quick_to_current
```

To see which features were selected:
```python
features = open("data/processed/version_E_selected_columns.txt").read().splitlines()
print(features)
```

---

## Split usage rules

| Split | When to use |
|-------|------------|
| `train_*.csv` | Fit the model |
| `val_*.csv` | Tune hyperparameters, early stopping, pick best epoch |
| `test_*.csv` | Report final metrics — use **once** at the very end only |

**Never fit scalers/transformers using val or test data.**
**Never retrain after seeing test scores.**

---

## Evaluation standard (Role 4)

All models evaluated on their respective `test_*.csv` with the same metrics:

```python
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score, confusion_matrix
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("AUC:      ", roc_auc_score(y_test, y_proba))
print("PR-AUC:   ", average_precision_score(y_test, y_proba))
print("F1:       ", f1_score(y_test, y_pred))
print("Recall:   ", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Primary metrics: Recall and PR-AUC.**
Missing a bankrupt company (false negative) is far more costly than a false alarm.

### Results table to fill in

| Model | Version | AUC | PR-AUC | F1 | Recall | Precision |
|-------|---------|-----|--------|----|--------|-----------|
| Logistic Regression | A or E | | | | | |
| Linear SVM | A | | | | | |
| Random Forest | B | | | | | |
| XGBoost | B | | | | | |
| LightGBM | B | | | | | |
| MLP | C | | | | | |
| MLP + PCA | D | | | | | |
| TabNet | D | | | | | |
| LogReg (engineered) | E | | | | | |

---

## Key numbers

| Item | Value |
|------|-------|
| Raw dataset | 6,819 × 96 |
| After cleaning | 6,819 × 95 (dropped `Net Income Flag`) |
| Train / Val / Test | 4,773 / 1,023 / 1,023 |
| Class imbalance | 3.23% bankrupt in all splits |
| Binary flag column | `Liability-Assets Flag` |
| Version C pruned | 20 cols dropped (|r|>0.95), 74 kept |
| Version D PCA | 94 → 50 components (95.46% variance) |
| Version E features | 16 (12 engineered + 4 original, all selected by target correlation) |
| random_state | 42 everywhere |
