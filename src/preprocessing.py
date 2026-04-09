"""
src/preprocessing.py
Reusable preprocessing functions for the bankruptcy-risk-ml project.
All functions follow the project rules:
  - random_state=42
  - Never fit before train/test split
  - Strip column name spaces on load
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

TARGET = "Bankrupt?"
RANDOM_STATE = 42


# ─── Loading ──────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """Load CSV and strip all column name spaces."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ─── Cleaning ─────────────────────────────────────────────────────────────────

def drop_zero_variance(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with zero variance. Returns (cleaned_df, dropped_cols)."""
    dropped = [c for c in df.columns if df[c].nunique() <= 1]
    return df.drop(columns=dropped), dropped


def get_binary_columns(df: pd.DataFrame, target: str = TARGET) -> list[str]:
    """Return list of columns (excl. target) with exactly 2 unique non-null values."""
    return sorted([
        c for c in df.columns
        if c != target and df[c].dropna().nunique() == 2
    ])


def winsorize(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99,
              cols: list[str] | None = None) -> pd.DataFrame:
    """
    Clip feature values at [lower, upper] quantiles computed on df.
    Call with train data only; apply the returned clip bounds to val/test manually
    using winsorize_with_bounds().
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=np.number).columns.tolist()
    bounds = {}
    for c in cols:
        lo = df[c].quantile(lower)
        hi = df[c].quantile(upper)
        df[c] = df[c].clip(lo, hi)
        bounds[c] = (lo, hi)
    return df, bounds


def winsorize_with_bounds(df: pd.DataFrame,
                          bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Apply pre-computed winsorization bounds to val/test data."""
    df = df.copy()
    for c, (lo, hi) in bounds.items():
        if c in df.columns:
            df[c] = df[c].clip(lo, hi)
    return df


# ─── Splitting ────────────────────────────────────────────────────────────────

def stratified_split(df: pd.DataFrame,
                     target: str = TARGET,
                     val_size: float = 0.15,
                     test_size: float = 0.15,
                     random_state: int = RANDOM_STATE
                     ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified 70/15/15 split (or custom proportions).
    Returns (train_df, val_df, test_df).
    """
    temp_size = val_size + test_size
    X, y = df.drop(columns=[target]), df[target]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=temp_size, stratify=y, random_state=random_state
    )
    relative_test = test_size / temp_size
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=relative_test, stratify=y_tmp, random_state=random_state
    )
    def reassemble(Xp, yp):
        d = Xp.copy(); d[target] = yp; return d
    return reassemble(X_tr, y_tr), reassemble(X_val, y_val), reassemble(X_te, y_te)


# ─── Scaling ──────────────────────────────────────────────────────────────────

def apply_standard_scaler(X_train: np.ndarray,
                           X_val: np.ndarray,
                           X_test: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on X_train; transform all three splits."""
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler


def apply_robust_scaler(X_train: np.ndarray,
                         X_val: np.ndarray,
                         X_test: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, RobustScaler]:
    """Fit RobustScaler on X_train; transform all three splits."""
    scaler = RobustScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler


# ─── Resampling ───────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: np.ndarray,
                k_neighbors: int = 5,
                random_state: int = RANDOM_STATE
                ) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE to training data only. Never call on val/test."""
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    return smote.fit_resample(X_train, y_train)


# ─── Feature selection ────────────────────────────────────────────────────────

def drop_high_correlation(X_train: np.ndarray, feature_names: list[str],
                           threshold: float = 0.95
                           ) -> tuple[list[str], list[str]]:
    """
    Identify features to drop based on pairwise correlation > threshold.
    Fit on X_train only. Returns (keep_cols, drop_cols).
    """
    df_corr = pd.DataFrame(X_train, columns=feature_names).corr().abs()
    upper = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))
    drop_cols = [c for c in upper.columns if any(upper[c] > threshold)]
    keep_cols = [c for c in feature_names if c not in drop_cols]
    return keep_cols, drop_cols


def apply_pca(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
              variance_threshold: float = 0.95,
              random_state: int = RANDOM_STATE
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, PCA]:
    """Fit PCA on X_train for variance_threshold; transform all three splits."""
    pca = PCA(n_components=variance_threshold, random_state=random_state)
    Xtr_pca = pca.fit_transform(X_train)
    Xv_pca  = pca.transform(X_val)
    Xte_pca = pca.transform(X_test)
    return Xtr_pca, Xv_pca, Xte_pca, pca


# ─── Utility ──────────────────────────────────────────────────────────────────

def arrays_to_df(X: np.ndarray, y: np.ndarray,
                 feature_names: list[str], target: str = TARGET) -> pd.DataFrame:
    """Combine feature array and target vector into a DataFrame."""
    df = pd.DataFrame(X, columns=feature_names)
    df[target] = y.astype(int)
    return df


def check_leakage(train_idx, val_idx, test_idx) -> bool:
    """Assert no index overlap between splits. Returns True if clean."""
    tr = set(train_idx); v = set(val_idx); te = set(test_idx)
    assert len(tr & v) == 0,  "LEAKAGE: train ∩ val is not empty"
    assert len(tr & te) == 0, "LEAKAGE: train ∩ test is not empty"
    assert len(v & te) == 0,  "LEAKAGE: val ∩ test is not empty"
    return True
