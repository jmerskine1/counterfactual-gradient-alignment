import os
import subprocess
import re
import yaml
import itertools
import json
import time
from datetime import datetime

# Expanded Experiment Configuration
LOSS_FUNCTIONS = [
    'multiclass_cross_entropy',
    'multiclass_split_loss_embedding_relu',
    'multiclass_split_loss_embedding_softplus',
    'multiclass_split_loss_embedding_sign'
]

# Focusing around 0.5
ALPHAS = [0.3, 0.5, 0.7] 
# Expanded Dataset Sizes
TRAIN_SIZES = [10, 20, 50, 100, 200]

CONFIG_PATH = 'Projects/SST-5/config.yaml'
RESULTS_FILE = 'Projects/SST-5/experiment_results_expanded.json'
MANIFEST_FILE = 'Projects/SST-5/experiment_manifest.txt'

def update_config(loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()

    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'init_samplesize: .*', f'init_samplesize: {train_size}', config_text)
    config_text = re.sub(r'active_sampling: .*', 'active_sampling: True', config_text)
    # Ensure visualisations are off
    config_text = re.sub(r'visualise: .*', 'visualise: false', config_text)
    config_text = re.sub(r'visualise_embeddings: .*', 'visualise_embeddings: false', config_text)
    # Set epochs to 5 for consistency
    config_text = re.sub(r'epochs: .*', 'epochs: 5', config_text)
    # Ensure dataset is correct
    config_text = re.sub(r'dataset: .*', 'dataset: integer_len32', config_text)

    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

def run_pipeline():
    try:
        result = subprocess.run(
            ['python3', 'Projects/SST-5/pipeline.py'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        matches_acc = re.findall(r'Validation | Loss: [\d\.]+, Accuracy: ([\d\.]+)%', output)
        accuracy = float(matches_acc[-1]) if matches_acc else None
        
        # Capture output filename
        matches_file = re.findall(r'Loading and saving to :  (MODEL_ENSEMBLE_.*)', output)
        output_file = matches_file[-1].strip() if matches_file else "Unknown"
        
        return accuracy, output_file
            
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        # print(e.stdout[-500:])
        # print(e.stderr[-500:])
        return None, "Failed"

def main():
    results = []
    
    ce_experiments = list(itertools.product(['multiclass_cross_entropy'], [0.0], TRAIN_SIZES))
    
    direction_experiments = list(itertools.product(
        [l for l in LOSS_FUNCTIONS if l != 'multiclass_cross_entropy'],
        ALPHAS,
        TRAIN_SIZES
    ))
    
    all_experiments = ce_experiments + direction_experiments
    
    print(f"Starting {len(all_experiments)} SST-5 experiments with 5 epochs...")
    
    with open(MANIFEST_FILE, 'w') as f:
        f.write("Timestamp, Loss Function, Alpha, Train Size, Accuracy, Output File\n")

    for loss_fn, alpha, train_size in all_experiments:
        print(f"\nRunning: Loss={loss_fn}, Alpha={alpha}, Size={train_size}")
        
        update_config(loss_fn, alpha, train_size)
        
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

    print("\n\n=== SST-5 Experiment Summary ===")
    print(f"{ 'Loss Function':<40} { 'Alpha':<10} { 'Size':<10} { 'Accuracy':<10}")
    print("-" * 75)
    for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{res['loss_function']:<40} {res['alpha']:<10} {res['train_size']:<10} {res['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
