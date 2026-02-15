import os
from main import run_all


def test_run_all_smoke(tmp_path):
    # Run a short pipeline and assert outputs are created
    run_all(data_path=str(tmp_path / "sim.csv"), days=60, lookback=5, seed=42, threshold=0.1, save_model_path=str(tmp_path / "model.joblib"), tune=False)
    assert os.path.exists(str(tmp_path / "sim.csv"))
    # predictions and metrics are written to working `data/` directory by design
    assert os.path.exists("data/predictions.csv")
    assert os.path.exists("data/metrics.json")
