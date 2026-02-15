import os
import joblib
from simulator import Simulator
from model import (
    create_features,
    train_and_predict,
    evaluate_metrics,
    load_model,
    feature_names,
    generate_rolling_features,
    compute_learning_curve,
    permutation_importances,
)


def test_create_features_shapes():
    sim = Simulator(seed=0)
    df = sim.simulate_days(10)
    X, y = create_features(df, lookback=3)
    # samples = len(df) - lookback - 1
    assert X.shape[0] == len(df) - 3 - 1
    assert X.shape[1] == 3 * 3  # liquidity_ratio, daily_withdrawals, loan_defaults
    assert y.shape[0] == X.shape[0]


def test_train_and_persist_and_load_model(tmp_path):
    sim = Simulator(seed=1)
    df = sim.simulate_days(30)
    model_path = tmp_path / "test_rf.joblib"

    model, X_test, y_test, preds, metrics = train_and_predict(df, lookback=4, save_model_path=str(model_path))
    assert os.path.exists(str(model_path))
    assert set(metrics.keys()).issuperset({"mae", "rmse", "r2"})

    loaded = load_model(str(model_path))
    preds2 = loaded.predict(X_test)
    assert preds2.shape == preds.shape


def test_evaluate_metrics_nonnegative():
    import numpy as np
    y_true = np.array([0.1, 0.2, 0.15])
    y_pred = np.array([0.11, 0.19, 0.14])
    m = evaluate_metrics(y_true, y_pred)
    assert m["mae"] >= 0
    assert m["rmse"] >= 0
    assert isinstance(m["r2"], float)


def test_feature_names_length():
    names = feature_names(lookback=4)
    assert len(names) == 4 * 3
    assert names[0].startswith("liquidity_ratio_t-")
    assert names[-1].startswith("loan_defaults_t-")


def test_generate_rolling_features_and_create_features_with_rolling():
    sim = Simulator(seed=0)
    df = sim.simulate_days(20)
    df2 = generate_rolling_features(df, windows=(3, 7))
    # check rolling cols exist
    assert "liquidity_ratio_roll_mean_3" in df2.columns
    assert "loan_defaults_roll_std_7" in df2.columns

    # create_features with rolling should append (len(windows) * metrics * stats) features
    lookback = 3
    X_plain, _ = create_features(df, lookback=lookback, use_rolling=False)
    X_roll, _ = create_features(df2, lookback=lookback, use_rolling=True, rolling_windows=(3, 7))
    base = lookback * 3
    appended = len((3, 7)) * 3 * 2  # windows * metrics(liquidity,daily_withdrawals,loan_defaults) * (mean,std)
    assert X_roll.shape[1] == base + appended
    assert X_plain.shape[0] == X_roll.shape[0]


def test_compute_learning_curve_and_permutation_importances():
    sim = Simulator(seed=1)
    df = sim.simulate_days(120)
    X, y = create_features(df, lookback=5)
    # simple split
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    sizes, train_rmses, val_rmses = compute_learning_curve(X_train, y_train, X_val, y_val, lambda: __import__("sklearn.ensemble").ensemble.RandomForestRegressor(n_estimators=10, random_state=0), train_sizes=(0.2, 0.5, 1.0))
    assert len(sizes) == 3
    assert len(train_rmses) == 3 and len(val_rmses) == 3
    assert all(r >= 0 for r in train_rmses)

    # train a small model and check permutation importances length
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=10, random_state=0)
    m.fit(X_train, y_train)
    imps = permutation_importances(m, X_val, y_val, n_repeats=5)
    assert len(imps) == X.shape[1]
