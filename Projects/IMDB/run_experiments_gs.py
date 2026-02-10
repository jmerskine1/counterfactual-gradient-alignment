import os, subprocess, re, yaml, itertools, json, time

CONFIG_PATH = 'Projects/IMDB/config.yaml'

def update_config(loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()
    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'init_samplesize: .*', f'init_samplesize: {train_size}', config_text)
    config_text = re.sub(r'active_sampling: .*', 'active_sampling: True', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 5', config_text)
    config_text = re.sub(r'n_models: .*', 'n_models: 1', config_text)
    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

results = []
for alpha, size in itertools.product([0.3, 0.5], [10, 100, 200]):
    print(f"Running IMDB GS: Alpha={alpha}, Size={size}")
    update_config('combined_loss_embedding_gs', alpha, size)
    res = subprocess.run(['python3', 'Projects/IMDB/new_pipeline.py'], capture_output=True, text=True)
    acc = re.findall(r'Validation \| Loss: [\d\.]+, Accuracy: ([\d\.]+)%', res.stdout)
    accuracy = float(acc[-1]) if acc else None
    results.append({'alpha': alpha, 'train_size': size, 'accuracy': accuracy})

with open('Projects/IMDB/experiment_results_gs.json', 'w') as f:
    json.dump(results, f, indent=4)