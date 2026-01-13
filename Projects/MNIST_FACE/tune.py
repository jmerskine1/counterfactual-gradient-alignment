import os
import subprocess
import yaml
import pickle
import itertools
from pathlib import Path
from multiprocessing import Pool, cpu_count


# === Paths ===
PROJECT_DIR = Path(__file__).parent
CONFIG_FILE = PROJECT_DIR / "config.yaml"
TRAIN_SCRIPT = PROJECT_DIR / "pipeline_gs_ensemble.py"   # <-- your big script
OUTPUT_DIR = PROJECT_DIR / "model_outputs"
os.makedirs(PROJECT_DIR / 'configs', exist_ok=True)
# === Search space ===
# search_space = {
#     "learning_rate": [1e-2, 1e-3, 1e-4],
#     "batch_size": [8,32, 64],
#     "embedding_size": [16,32, 64],}
#     "loss_function": ['multiclass_combined_loss_embedding', 
#                       'multiclass_gradient_supervision_embedding',
#                        'multiclass_cross_entropy']
# }
search_space = {}

data_space



def run_trial(trial_config):
    """Run one trial with a modified config.yaml and return validation accuracy."""
    # Create unique config file for this run
    trial_id = "_".join(f"{k}-{v}" for k, v in trial_config.items())
    trial_config_file = PROJECT_DIR / 'configs' / f"config_{trial_id}.yaml"

    # Load base config
    with open(CONFIG_FILE, "r") as f:
        config = yaml.unsafe_load(f)

    # Update with trial params
    config["hyperparams"].update(trial_config)
    with open(os.path.join('configs',trial_config_file), "w") as f:
        yaml.dump(config, f)

    # Run training with this config
    env = os.environ.copy()
    env["CONFIG_FILE"] = str(trial_config_file)  # if your train.py can read this
    try:
        subprocess.run(
            ["python", str(TRAIN_SCRIPT), "--config", str(trial_config_file)],
            env=env,
            check=True,
        )
    except subprocess.CalledProcessError:
        return (trial_config, None, "failed")

    # Find latest result
    data_name = config["data_params"]["dataset"]
    model_dir = OUTPUT_DIR / data_name
    latest_file = max(model_dir.glob("*.pkl"), key=os.path.getctime)
    print("GETTING:", latest_file)
    with open(latest_file, "rb") as f:
        results = pickle.load(f)
    val_acc = results["results"]["Validation"]["accuracy"][-1]

    return (trial_config, val_acc, latest_file.name)


def parallel_search():
    keys, values = zip(*search_space.items())
    all_combos = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    n_workers = min(4, cpu_count())  # Use up to 4 processes (M1 has efficiency/perf cores)
    print(f"Running {len(all_combos)} trials with {n_workers} workers...")

    with Pool(n_workers) as pool:
        results = pool.map(run_trial, all_combos)

    # Filter out failed runs
    results = [r for r in results if r[1] is not None]
    best = max(results, key=lambda x: x[1])

    print("\n=== BEST CONFIG ===")
    print(best[0], "→", best[1], "from", best[2])
    return results


if __name__ == "__main__":
    parallel_search()