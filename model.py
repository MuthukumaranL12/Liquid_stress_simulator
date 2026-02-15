import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from typing import Tuple, Dict, Any, Optional


def generate_rolling_features(df: pd.DataFrame, windows=(3, 7, 14)) -> pd.DataFrame:
    """Add rolling mean/std features for selected columns.

    Adds columns like `liquidity_ratio_roll_mean_3`, `daily_withdrawals_roll_std_7`, etc.
    Uses min_periods=1 so early rows are populated.
    """
    df = df.copy()
    cols = ["liquidity_ratio", "daily_withdrawals", "loan_defaults"]
    for w in windows:
        for c in cols:
            df[f"{c}_roll_mean_{w}"] = df[c].rolling(window=w, min_periods=1).mean()
            df[f"{c}_roll_std_{w}"] = df[c].rolling(window=w, min_periods=1).std().fillna(0.0)
    return df


def create_features(df: pd.DataFrame, lookback: int = 5, use_rolling: bool = False, rolling_windows=(3, 7, 14)) -> Tuple[np.ndarray, np.ndarray]:
    """Create lag features from simulated dataframe.

    Features (base): flattened liquidity_ratio, daily_withdrawals, loan_defaults for the previous `lookback` days.
    If `use_rolling` is True, additional rolling mean/std features (for windows in `rolling_windows`) are appended
    using the most recent row in the lookback window.

    Target: next-day liquidity_ratio.
    """
    if use_rolling:
        df = generate_rolling_features(df, windows=rolling_windows)

    X = []
    y = []
    for i in range(lookback, len(df) - 1):
        window = df.iloc[i - lookback:i]
        feats = []
        # flatten liquidity_ratio, daily_withdrawals, loan_defaults for past days
        feats.extend(window['liquidity_ratio'].values.tolist())
        feats.extend(window['daily_withdrawals'].values.tolist())
        feats.extend(window['loan_defaults'].values.tolist())

        # append rolling statistics from the most recent day in the window
        if use_rolling:
            last_row = window.iloc[-1]
            for w in rolling_windows:
                for c in ['liquidity_ratio', 'daily_withdrawals', 'loan_defaults']:
                    feats.append(float(last_row[f"{c}_roll_mean_{w}"]))
                    feats.append(float(last_row[f"{c}_roll_std_{w}"]))

        X.append(feats)
        # target is next day liquidity_ratio
        y.append(float(df.iloc[i + 1]['liquidity_ratio']))

    X = np.array(X)
    y = np.array(y)
    return X, y


def compute_learning_curve(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
                           model_ctor, train_sizes=(0.2, 0.4, 0.6, 0.8, 1.0)) -> Tuple[list, list, list]:
    """Compute a simple learning curve (train RMSE vs val RMSE) for increasing training set sizes.

    model_ctor: callable that returns a fresh estimator when invoked (e.g., lambda: RandomForestRegressor(...)).
    Returns: (sizes, train_rmse_list, val_rmse_list)
    """
    sizes = []
    train_rmses = []
    val_rmses = []
    n = len(X_train)
    for frac in train_sizes:
        m = int(max(1, frac * n))
        sizes.append(m)
        model = model_ctor()
        model.fit(X_train[:m], y_train[:m])
        train_pred = model.predict(X_train[:m])
        val_pred = model.predict(X_val)
        train_rmses.append(float(mean_squared_error(y_train[:m], train_pred) ** 0.5))
        val_rmses.append(float(mean_squared_error(y_val, val_pred) ** 0.5))
    return sizes, train_rmses, val_rmses


def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return y_true - y_pred


from sklearn.inspection import permutation_importance


def permutation_importances(model, X: np.ndarray, y: np.ndarray, n_repeats: int = 10):
    """Return a DataFrame with permutation importances (fallback / complement to SHAP)."""
    r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=0)
    importances = r.importances_mean
    return importances



def feature_names(lookback: int = 5) -> list:
    """Return human-readable feature names for the flattened feature vector.

    Order matches `create_features`: liquidity_ratio lags, daily_withdrawals lags, loan_defaults lags
    (from oldest to most recent: t-lookback ... t-1).
    """
    names = []
    # liquidity_ratio lags: t-lookback ... t-1
    for lag in range(lookback, 0, -1):
        names.append(f"liquidity_ratio_t-{lag}")
    # daily_withdrawals lags
    for lag in range(lookback, 0, -1):
        names.append(f"daily_withdrawals_t-{lag}")
    # loan_defaults lags
    for lag in range(lookback, 0, -1):
        names.append(f"loan_defaults_t-{lag}")
    return names


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(mse ** 0.5)
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_and_predict(df: pd.DataFrame,
                      lookback: int = 5,
                      test_size: float = 0.2,
                      random_state: int = 42,
                      save_model_path: Optional[str] = None,
                      tune: bool = False) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train RandomForest and return model, X_test, y_test, preds and regression metrics.

    If save_model_path is provided the trained model is persisted to disk.
    If tune=True a small grid search will run (keeps default fast).
    """
    X, y = create_features(df, lookback=lookback)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    if tune:
        param_grid = {"n_estimators": [50, 100], "max_depth": [None, 5, 10]}
        g = GridSearchCV(RandomForestRegressor(random_state=random_state), param_grid, cv=3, scoring='neg_mean_squared_error')
        g.fit(X_train, y_train)
        model = g.best_estimator_
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = evaluate_metrics(y_test, preds)

    if save_model_path:
        joblib.dump(model, save_model_path)

    return model, X_test, y_test, preds, metrics


def load_model(path: str):
    return joblib.load(path)


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/simulated_data.csv")
    model, X_test, y_test, preds, metrics = train_and_predict(df)
    print("Test samples:", len(y_test))
    print(metrics)
