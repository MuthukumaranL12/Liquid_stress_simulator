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

Development & testing

- Run the test suite: `pytest -q`
- CI workflow (GitHub Actions) runs tests and a smoke pipeline run on push/PR

Next steps (ideas)

- Improve feature engineering and hyperparameter search
- Add more realistic stress scenarios and scenario tests
- Build a dashboard for alerts

