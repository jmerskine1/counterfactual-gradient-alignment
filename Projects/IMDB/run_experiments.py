import os
import subprocess
import re
import yaml
import itertools
import json
import time
from datetime import datetime

# IMDB Experiment Configuration (Binary)
LOSS_FUNCTIONS = [
    'cross_entropy',
    'combined_loss_embedding_relu',
    'combined_loss_embedding_softplus',
    'combined_loss_embedding_sign'
]

ALPHAS = [0.5] 
TRAIN_SIZES = [10, 100, 200]

CONFIG_PATH = 'Projects/IMDB/config.yaml'
RESULTS_FILE = 'Projects/IMDB/experiment_results_no_active.json'
MANIFEST_FILE = 'Projects/IMDB/experiment_manifest_no_active.txt'

def update_config(loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()

    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    
    # In IMDB new_pipeline.py, it seems to hardcode init_samplesize = 600 if active_sampling is True.
    # We might need to override active_sampling to False and control train_size differently if pipeline allows.
    # Actually, looking at new_pipeline.py:
    # if active_sampling:
    #     init_samplesize = 600
    # ...
    # else:
    #     training_set = full_training_set
    
    # We should probably set active_sampling: False and let it use the full training set?
    # But then we need to subset the training set ourselves in the config or pipeline.
    # Wait, the pipeline doesn't seem to have a 'train_size' subsetting logic if active_sampling is False.
    
    # I will modify the pipeline locally to support 'train_size' if needed, 
    # but first let's see if I can just use init_samplesize by changing the code.
    
    config_text = re.sub(r'active_sampling: .*', 'active_sampling: False', config_text)
    # We'll need to patch the pipeline to use init_samplesize from config.
    
    config_text = re.sub(r'visualise: .*', 'visualise: false', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 5', config_text)
    config_text = re.sub(r'n_models: .*', 'n_models: 1', config_text)

    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

def run_pipeline():
    try:
        result = subprocess.run(
            ['python3', 'Projects/IMDB/new_pipeline.py'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        # IMDB prints "Validation | Loss: ..., Accuracy: ...%" ? 
        # Looking at new_pipeline.py, it uses generate_results_ensemble.
        matches_acc = re.findall(r'Validation | Loss: [\d\.]+, Accuracy: ([\d\.]+)%', output)
        accuracy = float(matches_acc[-1]) if matches_acc else None
        
        matches_file = re.findall(r'saving: (MODEL_ENSEMBLE_.*)', output)
        output_file = matches_file[-1].strip() if matches_file else "Unknown"
        
        return accuracy, output_file
            
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        return None, "Failed"

def main():
    # Patch IMDB pipeline to respect init_samplesize from config
    with open('Projects/IMDB/new_pipeline.py', 'r') as f:
        pipeline_code = f.read()
    
    if 'init_samplesize = 600' in pipeline_code:
        pipeline_code = pipeline_code.replace('init_samplesize = 600', "init_samplesize = config['data_params'].get('init_samplesize', 600)")
        with open('Projects/IMDB/new_pipeline.py', 'w') as f:
            f.write(pipeline_code)
        print("Patched IMDB pipeline to respect init_samplesize.")

    results = []
    
    ce_experiments = list(itertools.product(['cross_entropy'], [0.0], TRAIN_SIZES))
    direction_experiments = list(itertools.product(
        [l for l in LOSS_FUNCTIONS if l != 'cross_entropy'],
        ALPHAS,
        TRAIN_SIZES
    ))
    
    all_experiments = ce_experiments + direction_experiments
    
    print(f"Starting {len(all_experiments)} IMDB experiments...")
    
    with open(MANIFEST_FILE, 'w') as f:
        f.write("Timestamp, Loss Function, Alpha, Train Size, Accuracy, Output File\n")

    for loss_fn, alpha, train_size in all_experiments:
        print(f"\nRunning: Loss={loss_fn}, Alpha={alpha}, Size={train_size}")
        
        update_config(loss_fn, alpha, train_size)
        
        # Also need to add init_samplesize to config if not present
        with open(CONFIG_PATH, 'r') as f:
            conf = yaml.unsafe_load(f)
        conf['data_params']['init_samplesize'] = train_size
        with open(CONFIG_PATH, 'w') as f:
            yaml.dump(conf, f)

        start_time = time.time()
        accuracy, output_file = run_pipeline()
        duration = time.time() - start_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if accuracy is not None:
            print(f"  -> Accuracy: {accuracy}% (Time: {duration:.2f}s)")
            result_entry = {
                'timestamp': timestamp,
                'loss_function': loss_fn,
                'alpha': alpha,
                'train_size': train_size,
                'accuracy': accuracy,
                'duration': duration,
                'output_file': output_file
            }
            results.append(result_entry)
            with open(MANIFEST_FILE, 'a') as f:
                f.write(f"{timestamp}, {loss_fn}, {alpha}, {train_size}, {accuracy}, {output_file}\n")
        else:
            print("  -> Failed.")
            
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    print("\n\n=== IMDB Experiment Summary ===")
    print(f"{'Loss Function':<40} {'Alpha':<10} {'Size':<10} {'Accuracy':<10}")
    print("-" * 75)
    for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{res['loss_function']:<40} {res['alpha']:<10} {res['train_size']:<10} {res['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
