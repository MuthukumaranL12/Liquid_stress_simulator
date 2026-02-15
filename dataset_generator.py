import os
from simulator import Simulator


def generate_and_save(save_path="data/simulated_data.csv", days=365, seed=None):
    dirname = os.path.dirname(save_path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    sim = Simulator(seed=seed)
    df = sim.simulate_days(days)
    df.to_csv(save_path, index=False)
    return df


if __name__ == "__main__":
    df = generate_and_save("data/simulated_data.csv", days=365, seed=42)
    print(df.tail())
