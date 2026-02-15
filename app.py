import streamlit as st
import pandas as pd
import io
import os
import matplotlib.pyplot as plt

from dataset_generator import generate_and_save
from model import (
    create_features,
    evaluate_metrics,
    feature_names,
    generate_rolling_features,
    compute_learning_curve,
    compute_residuals,
    permutation_importances,
)
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from visualizer import plot_actual_vs_pred

st.set_page_config(page_title="Liquidity Stress Simulator", layout="wide")

st.title("Liquidity Stress Simulator — Dashboard")

# Attempt to load pre-trained artifacts (deployment-friendly behavior)
_pretrained_df = None
_pretrained_model = None
try:
    if os.path.exists("data/simulated_data.csv"):
        _pretrained_df = pd.read_csv("data/simulated_data.csv")
except Exception:
    _pretrained_df = None

try:
    if os.path.exists("models/rf_model.joblib"):
        _pretrained_model = joblib.load("models/rf_model.joblib")
except Exception:
    _pretrained_model = None

with st.sidebar:
    st.header("Run / Settings")
    days = st.number_input("Days to simulate", min_value=30, max_value=2000, value=365)
    seed = st.number_input("Random seed", value=42)
    lookback = st.number_input("Lookback (days)", min_value=1, max_value=30, value=5)
    threshold = st.number_input("Crisis threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    use_rolling = st.checkbox("Use rolling features (mean/std)")
    show_learning = st.checkbox("Show learning curve")
    show_residuals = st.checkbox("Show residuals")
    compare = st.checkbox("Compare with LinearRegression baseline")
    show_importances = st.checkbox("Show feature importances")
    compute_shap = st.checkbox("Compute SHAP explanations (optional)")
    tune = st.checkbox("Run small hyperparameter tune (slower)")

    if _pretrained_model is not None:
        st.success("Pre-trained model found (loaded)")
        st.text("Model features: %d" % getattr(_pretrained_model, "n_features_in_", -1))
    else:
        st.warning("No pre-trained model found in models/ — press 'Run simulation & train model' to create one")

    run_btn = st.button("Run simulation & train model")

st.markdown("---")

if run_btn:
    # retrain (user-triggered) — identical to previous behaviour
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    with st.spinner("Simulating data..."):
        df = generate_and_save(save_path="data/simulated_data.csv", days=int(days), seed=int(seed))
    st.success("Simulation complete")

    st.subheader("Simulated data (head)")
    st.dataframe(df.head())

    # Optionally augment df with rolling features for display
    if use_rolling:
        df = generate_rolling_features(df, windows=(3, 7, 14))

    # Build features & split (create_features handles rolling if requested)
    X, y = create_features(df, lookback=int(lookback), use_rolling=use_rolling, rolling_windows=(3, 7, 14))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    with st.spinner("Training RandomForest..."):
        rf = RandomForestRegressor(n_estimators=100, random_state=int(seed))
        rf.fit(X_train, y_train)
        preds_rf = rf.predict(X_test)
        metrics_rf = evaluate_metrics(y_test, preds_rf)
        joblib.dump(rf, "models/rf_model.joblib")
    st.success("RandomForest trained and persisted to models/")

    # Learning curve
    if show_learning:
        with st.spinner("Computing learning curve..."):
            sizes, train_rmses, val_rmses = compute_learning_curve(
                X_train, y_train, X_test, y_test, lambda: RandomForestRegressor(n_estimators=100, random_state=int(seed)),
                train_sizes=(0.2, 0.4, 0.6, 0.8, 1.0),
            )
        lc_df = pd.DataFrame({"train_rmse": train_rmses, "val_rmse": val_rmses}, index=sizes)
        st.subheader("Learning curve (RMSE)")
        st.line_chart(lc_df)

    # Residuals
    if show_residuals:
        res = compute_residuals(y_test, preds_rf)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].scatter(preds_rf, res, alpha=0.6)
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residual")
        axes[0].set_title("Predicted vs Residual")

        axes[1].hist(res, bins=30)
        axes[1].set_title("Residual distribution")
        st.subheader("Residual diagnostics")
        st.pyplot(fig)

    compare_metrics = None
    if compare:
        with st.spinner("Training baseline (LinearRegression)..."):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            preds_lr = lr.predict(X_test)
            metrics_lr = evaluate_metrics(y_test, preds_lr)
        compare_metrics = {"RandomForest": metrics_rf, "LinearRegression": metrics_lr}

    # Show metrics
    st.subheader("Regression metrics — RandomForest")
    st.json(metrics_rf)

    if compare_metrics:
        st.subheader("Model comparison")
        cmp_df = pd.DataFrame(compare_metrics).T
        st.dataframe(cmp_df)

    # Predictions table
    indices = list(range(int(lookback), int(lookback) + len(preds_rf)))
    preds_df = pd.DataFrame({"index": indices, "predicted_liquidity": preds_rf, "actual_liquidity": y_test})

    st.subheader("Predictions (test set)")
    st.dataframe(preds_df)

    # Alerts
    alerts = preds_df[preds_df["predicted_liquidity"] < threshold]
    st.subheader(f"Alerts (predicted < {threshold}) — {len(alerts)}")
    if not alerts.empty:
        st.dataframe(alerts)
    else:
        st.info("No alerts for the chosen threshold")

    # Plot
    st.subheader("Actual vs Predicted")
    plot_actual_vs_pred(preds_df["index"].values, preds_df["actual_liquidity"].values, preds_df["predicted_liquidity"].values, threshold=threshold, save_path=None)
    st.pyplot(plt)

    # Feature importances (same as before)
    if show_importances:
        fnames = feature_names(int(lookback))
        if use_rolling:
            for w in (3, 7, 14):
                for c in ['liquidity_ratio', 'daily_withdrawals', 'loan_defaults']:
                    fnames.append(f"{c}_roll_mean_{w}")
                    fnames.append(f"{c}_roll_std_{w}")

        importances = rf.feature_importances_
        imp_df = pd.DataFrame({"feature": fnames, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
        st.subheader("Feature importances (RandomForest)")
        st.dataframe(imp_df.head(20))
        st.bar_chart(imp_df.set_index("feature")["importance"])

        if compute_shap:
            try:
                import shap

                st.subheader("SHAP summary (approx)")
                explainer = shap.Explainer(rf)
                sample = X_test if X_test.shape[0] <= 200 else X_test[:200]
                shap_values = explainer(sample)
                shap.summary_plot(shap_values.values, sample, feature_names=fnames, show=False)
                st.pyplot(plt.gcf())
            except Exception as e:
                st.warning("SHAP is not available or failed to run — install `shap` to enable explanations.\n" + str(e))
                imps = permutation_importances(rf, X_test, y_test)
                p_imp_df = pd.DataFrame({"feature": fnames, "perm_importance": imps})
                p_imp_df = p_imp_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)
                st.subheader("Permutation importances (fallback)")
                st.dataframe(p_imp_df.head(20))
        else:
            imps = permutation_importances(rf, X_test, y_test)
            p_imp_df = pd.DataFrame({"feature": fnames, "perm_importance": imps})
            p_imp_df = p_imp_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)
            st.subheader("Permutation importances (complementary)")
            st.dataframe(p_imp_df.head(20))

    csv_bytes = preds_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")

    if os.path.exists("models/rf_model.joblib"):
        with open("models/rf_model.joblib", "rb") as f:
            model_bytes = f.read()
        st.download_button("Download trained model (joblib)", data=model_bytes, file_name="rf_model.joblib", mime="application/octet-stream")

else:
    # If there is a committed pre-trained model + dataset, show results immediately (deployment-friendly)
    if _pretrained_model is not None and _pretrained_df is not None:
        df = _pretrained_df.copy()
        st.subheader("Using committed dataset + pre-trained model (read-only)")

        if use_rolling:
            df = generate_rolling_features(df, windows=(3, 7, 14))

        # create features and ensure compatibility with loaded model
        try:
            X, y = create_features(df, lookback=int(lookback), use_rolling=use_rolling, rolling_windows=(3, 7, 14))
        except Exception as e:
            st.error(f"Failed to create features from committed dataset: {e}")
            X, y = None, None

        if X is not None:
            # check feature-length compatibility
            expected = getattr(_pretrained_model, "n_features_in_", None)
            if expected is not None and X.shape[1] != expected:
                st.warning(f"Pre-trained model expects {expected} features but current lookback produces {X.shape[1]} features. Set lookback to match or retrain the model.")
            else:
                # compute predictions on the test split so UI mirrors training output
                from sklearn.model_selection import train_test_split

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                preds = _pretrained_model.predict(X_test)
                metrics = evaluate_metrics(y_test, preds)

                st.subheader("Regression metrics (committed model)")
                st.json(metrics)

                indices = list(range(int(lookback), int(lookback) + len(preds)))
                preds_df = pd.DataFrame({"index": indices, "predicted_liquidity": preds, "actual_liquidity": y_test})
                st.subheader("Predictions (test set)")
                st.dataframe(preds_df)

                st.subheader("Actual vs Predicted")
                plot_actual_vs_pred(preds_df["index"].values, preds_df["actual_liquidity"].values, preds_df["predicted_liquidity"].values, threshold=threshold, save_path=None)
                st.pyplot(plt)

                if show_importances:
                    fnames = feature_names(int(lookback))
                    if use_rolling:
                        for w in (3, 7, 14):
                            for c in ['liquidity_ratio', 'daily_withdrawals', 'loan_defaults']:
                                fnames.append(f"{c}_roll_mean_{w}")
                                fnames.append(f"{c}_roll_std_{w}")

                    try:
                        importances = _pretrained_model.feature_importances_
                        imp_df = pd.DataFrame({"feature": fnames, "importance": importances})
                        imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
                        st.subheader("Feature importances (committed RandomForest)")
                        st.dataframe(imp_df.head(20))
                        st.bar_chart(imp_df.set_index("feature")["importance"])
                    except Exception:
                        st.info("Committed model does not expose feature importances")

    else:
        st.info("Configure parameters in the sidebar and click 'Run simulation & train model'.")

# st.markdown("---")
# st.caption("Tip: run `streamlit run app.py` to start the dashboard locally.")

# if __name__ == "__main__":
#     # allows running `python app.py` but Streamlit recommended: `streamlit run app.py`
#     st.write("Start the app with: `streamlit run app.py`")
