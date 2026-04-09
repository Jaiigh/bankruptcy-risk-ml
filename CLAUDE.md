# bankruptcy-risk-ml

## Project Context

- Dataset: data/raw/data.csv (6819 rows, 96 cols, financial ratios)
- Target: `Bankrupt?` (binary, heavily imbalanced ~3.2% positive)
- All column names have leading spaces — always strip them first
- Zero-variance column: `Net Income Flag` — always drop it
- Never apply any fitting (scaler, SMOTE, transformer) before train/test split

## Coding Standards

- Python 3.10+, pandas, scikit-learn, imbalanced-learn
- All reusable functions go in src/preprocessing.py
- Notebooks prefixed with numbers: 01*, 02*, etc.
- random_state=42 everywhere
