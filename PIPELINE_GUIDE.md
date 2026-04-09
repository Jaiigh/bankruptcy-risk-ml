# Pipeline Guide — How to Use the Processed Data

## Which version to load per model

| Version | File prefix | Scaler | Resampling | Features | Use for |
|---------|------------|--------|-----------|---------|---------|
| A | `outputs/version_A_*.csv` | StandardScaler | None | 94 | Logistic Regression, Linear SVM |
| B | `outputs/version_B_*.csv` | RobustScaler | SMOTE | 94 | XGBoost, Random Forest, LightGBM |
| C | `outputs/version_C_*.csv` | StandardScaler | None | 74 | MLP (correlation-pruned features) |
| D | `outputs/version_D_*.csv` | StandardScaler + PCA | None | 50 (PC1…PC50) | MLP, TabNet |

---

## Standard loading pattern (same for every version)

```python
import pandas as pd

train = pd.read_csv("outputs/version_X_train.csv")  # replace X with A/B/C/D
val   = pd.read_csv("outputs/version_X_val.csv")
test  = pd.read_csv("outputs/version_X_test.csv")

X_train = train.drop(columns=["Bankrupt?"])
y_train = train["Bankrupt?"]

X_val   = val.drop(columns=["Bankrupt?"])
y_val   = val["Bankrupt?"]

X_test  = test.drop(columns=["Bankrupt?"])
y_test  = test["Bankrupt?"]
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
- Use `class_weight='balanced'` — Version A keeps the natural 3.23% imbalance
- Do NOT use SMOTE on top — it was deliberately excluded for this version

### Version B → XGBoost, Random Forest, LightGBM
```python
from xgboost import XGBClassifier

train = pd.read_csv("outputs/version_B_train.csv")
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

model = XGBClassifier(random_state=42)
model.fit(X_train, y_train)
```
- No `class_weight` needed — SMOTE already balanced train to 50/50
- Val and test are NOT resampled — they reflect the real 3.23% distribution

---

## Role 3 — Deep Learning

### Version C → MLP with named features
```python
train = pd.read_csv("outputs/version_C_train.csv")  # 74 named financial ratio columns
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

# Feed directly into MLP — correlated noise already removed
# Use class_weight or weighted loss to handle 3.23% imbalance
```

### Version D → MLP / TabNet with PCA input
```python
train = pd.read_csv("outputs/version_D_train.csv")  # 50 columns: PC1, PC2, ... PC50
X_train, y_train = train.drop(columns=["Bankrupt?"]), train["Bankrupt?"]

# Columns are principal components, not original feature names
# 95.46% of original variance is retained across 50 components
```

---

## Split usage rules (applies to all roles)

| Split | When to use |
|-------|------------|
| `train_*.csv` | Fit the model |
| `val_*.csv` | Tune hyperparameters, pick best epoch, early stopping |
| `test_*.csv` | Report final metrics — touch this ONCE at the very end |

**Never fit any scaler or transformer using val or test data.**
**Never retrain after evaluating on test.**

---

## Evaluation standard (Role 4)

All models must report the same metrics on their respective `test_*.csv`:

```python
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, average_precision_score, confusion_matrix
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("AUC:    ", roc_auc_score(y_test, y_proba))
print("PR-AUC: ", average_precision_score(y_test, y_proba))
print("F1:     ", f1_score(y_test, y_pred))
print("Recall: ", recall_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

**Primary metrics: Recall and PR-AUC.**
Missing a bankrupt company (false negative) is far more costly than a false alarm.

### Results table format

| Model | Version | AUC | PR-AUC | F1 | Recall | Precision |
|-------|---------|-----|--------|----|--------|-----------|
| Logistic Regression | A | | | | | |
| Linear SVM | A | | | | | |
| Random Forest | B | | | | | |
| XGBoost | B | | | | | |
| LightGBM | B | | | | | |
| MLP | C | | | | | |
| MLP + PCA | D | | | | | |
| TabNet | D | | | | | |

---

## Key numbers to remember

- **Original dataset:** 6,819 rows × 96 columns
- **After cleaning:** 6,819 rows × 95 columns (dropped `Net Income Flag`)
- **Train / Val / Test:** 4,773 / 1,023 / 1,023 rows
- **Class imbalance:** 3.23% bankrupt in all splits
- **Binary flag column:** `Liability-Assets Flag` (no scaling needed)
- **Version C pruned:** 20 columns dropped (|r| > 0.95), 74 kept
- **Version D PCA:** 94 features → 50 components (95.46% variance)
- **random_state=42** everywhere
