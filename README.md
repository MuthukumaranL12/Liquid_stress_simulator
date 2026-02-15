# Liquidity Stress Detector

Simple bank liquidity simulator + dataset generator, Random Forest model to predict next-day liquidity ratio, visualization and evaluation.

Quick start

1) Create a Python environment and install pinned dependencies:

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```

2) Run the pipeline (CLI):

```bash
python main.py --days 365 --seed 42 --lookback 5 --threshold 0.1
```

3) Run the Streamlit dashboard (interactive):

```bash
streamlit run app.py
```

What the run produces

- `data/simulated_data.csv` — simulated time series input
- `data/predictions.csv` — model predictions vs actuals
- `data/liquidity_plot.png` — visualization
- `data/metrics.json` — regression + detection metrics
- `models/rf_model.joblib` — persisted trained model

Deploy to Streamlit Cloud (recommended)

1) Push this repository to GitHub (create repo if needed). The repo must include `app.py` and `requirements.txt` at the top level and the committed `models/rf_model.joblib` + `data/simulated_data.csv` files.

2) Go to https://share.streamlit.io and log in. Click **New app** → choose your GitHub repo/branch and set `app.py` as the entrypoint.

3) Enable **Auto deploy** in the Streamlit app settings so the app refreshes whenever you push to the selected branch.

Automatic artifact refresh (optional)

- There is a GitHub Actions workflow `.github/workflows/generate_artifacts.yml` which can re-run `train_model.py` on a schedule or by manual dispatch and commit updated `models/` and `data/` back to `main`.
- This workflow runs only on manual dispatch and on a daily schedule to avoid push loops.

Development & testing

- Run the test suite: `pytest -q`
- CI workflow (GitHub Actions) runs tests and a smoke pipeline run on push/PR

Next steps (ideas)

- Improve feature engineering and hyperparameter search
- Add more realistic stress scenarios and scenario tests
- Build a dashboard for alerts

