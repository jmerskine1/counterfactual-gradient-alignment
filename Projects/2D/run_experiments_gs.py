import os, subprocess, re, yaml, itertools, json, time

CONFIG_PATH = 'Projects/2D/config.yaml'

def update_config(dataset, loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()
    config_text = re.sub(r'dataset: .*', f'dataset: {dataset}', config_text)
    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'train_size: .*', f'train_size: {train_size}', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 10', config_text)
    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

results = []
for dataset, size in itertools.product(['XOR', 'Circles', 'TwoMoons'], [10, 30, 100]):
    print(f"Running 2D GS: Dataset={dataset}, Size={size}")
    update_config(dataset, 'multiclass_combined_gs', 0.5, size)
    res = subprocess.run(['python3', 'Projects/2D/pipeline_gs_ensemble_2D.py'], capture_output=True, text=True)
    acc = re.findall(r'Validation \| Loss: [\d\.]+, Accuracy: ([\d\.]+)%', res.stdout)
    accuracy = float(acc[-1]) if acc else None
    results.append({'dataset': dataset, 'train_size': size, 'accuracy': accuracy})

with open('Projects/2D/experiment_results_gs.json', 'w') as f:
    json.dump(results, f, indent=4)