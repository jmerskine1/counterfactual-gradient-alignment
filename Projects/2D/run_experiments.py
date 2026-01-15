import os
import subprocess
import re
import yaml
import itertools
import json
import time
from datetime import datetime

# 2D Experiment Configuration
LOSS_FUNCTIONS = [
    'multiclass_cross_entropy',
    'multiclass_combined_loss_embedding_relu',
    'multiclass_combined_loss_embedding_softplus',
    'multiclass_combined_loss_embedding_sign'
]
# Note: For 2D, we might want 'multiclass_combined_loss' if it uses 'direction'.
# But I implemented 'multiclass_split_loss_embedding_relu' etc. which assume embeddings.
# The 2D models are simple classifiers (MLP?), they output logits directly from features?
# 2D inputs are just coordinates. There is no "embedding layer" usually.
# So 'multiclass_direction_relu' (using gradients on input) might be better?
# But gradients on input requires input to be differentiable. X is coords (float), so it works.
# Let's check 'multiclass_direction_relu' implementation. 
# It takes (params, model, batch...). It computes gradients w.r.t input X.
# But 'multiclass_split_loss_embedding_relu' assumes input is embedding.
# So we should use a 'multiclass_combined_loss_relu' that wraps 'multiclass_direction_relu' directly on X.

# I need to implement 'multiclass_combined_loss_relu' etc. in loss_functions.py first?
# 'multiclass_combined_loss' uses 'multiclass_direction'.
# I can just use 'multiclass_combined_loss' and change 'multiclass_direction' behavior? No.
# I should add 'multiclass_combined_loss_relu' to loss_functions.py.

# Wait, I'll stick to 'multiclass_combined_loss' for now if I can't easily add more.
# But I promised to test variants.
# So I must add 'multiclass_combined_loss_relu' etc. to loss_functions.py.

# ALPHAS = [0.5] 
# DATASETS = ['XOR', 'Circles', 'TwoMoons']
# TRAIN_SIZES = [10, 30, 50, 100]

CONFIG_PATH = 'Projects/2D/config.yaml'
RESULTS_FILE = 'Projects/2D/experiment_results_expanded.json'
MANIFEST_FILE = 'Projects/2D/experiment_manifest.txt'

def update_config(dataset, loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()

    config_text = re.sub(r'dataset: .*', f'dataset: {dataset}', config_text)
    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'train_size: .*', f'train_size: {train_size}', config_text)
    
    config_text = re.sub(r'visualise: .*', 'visualise: False', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 10', config_text) # Faster

    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

def run_pipeline():
    try:
        result = subprocess.run(
            ['python3', 'Projects/2D/pipeline_gs_ensemble_2D.py'],
            capture_output=True,
            text=True,
            check=True
        )
        output = result.stdout
        
        matches_acc = re.findall(r'Validation | Loss: [\d\.]+, Accuracy: ([\d\.]+)%', output)
        accuracy = float(matches_acc[-1]) if matches_acc else None
        
        matches_file = re.findall(r'Loading and saving to :  (MODEL_ENSEMBLE_.*)', output)
        output_file = matches_file[-1].strip() if matches_file else "Unknown"
        
        return accuracy, output_file
            
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed: {e}")
        return None, "Failed"

def main():
    results = []
    
    # We need to define LOSS_FUNCTIONS that are actually available.
    # I'll use 'multiclass_combined_loss' as a placeholder for "Directional" 
    # but I really need the variants.
    # I will assume I added them (I need to do that next).
    
    LOSS_FUNCTIONS = [
        'multiclass_cross_entropy',
        'multiclass_combined_loss_relu',
        'multiclass_combined_loss_softplus',
        'multiclass_combined_loss_sign'
    ]
    DATASETS = ['XOR', 'Circles', 'TwoMoons']
    TRAIN_SIZES = [10, 30, 100]
    ALPHAS = [0.5]

    all_experiments = []
    for d in DATASETS:
        all_experiments.extend(list(itertools.product([d], ['multiclass_cross_entropy'], [0.0], TRAIN_SIZES)))
        all_experiments.extend(list(itertools.product([d], [l for l in LOSS_FUNCTIONS if l != 'multiclass_cross_entropy'], ALPHAS, TRAIN_SIZES)))
    
    print(f"Starting {len(all_experiments)} 2D experiments...")
    
    with open(MANIFEST_FILE, 'w') as f:
        f.write("Timestamp, Dataset, Loss Function, Alpha, Train Size, Accuracy, Output File\n")

    for dataset, loss_fn, alpha, train_size in all_experiments:
        print(f"\nRunning: Dataset={dataset}, Loss={loss_fn}, Alpha={alpha}, Size={train_size}")
        
        update_config(dataset, loss_fn, alpha, train_size)
        
        start_time = time.time()
        accuracy, output_file = run_pipeline()
        duration = time.time() - start_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if accuracy is not None:
            print(f"  -> Accuracy: {accuracy}% (Time: {duration:.2f}s)")
            
            result_entry = {
                'timestamp': timestamp,
                'dataset': dataset,
                'loss_function': loss_fn,
                'alpha': alpha,
                'train_size': train_size,
                'accuracy': accuracy,
                'duration': duration,
                'output_file': output_file
            }
            results.append(result_entry)
            
            with open(MANIFEST_FILE, 'a') as f:
                f.write(f"{timestamp}, {dataset}, {loss_fn}, {alpha}, {train_size}, {accuracy}, {output_file}\n")
        else:
            print("  -> Failed.")
            
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    print("\n\n=== 2D Experiment Summary ===")
    print(f"{ 'Dataset':<10} { 'Loss Function':<40} { 'Alpha':<10} { 'Size':<10} { 'Accuracy':<10}")
    print("-" * 85)
    for res in sorted(results, key=lambda x: (x['dataset'], x['accuracy']), reverse=True):
        print(f"{res['dataset']:<10} {res['loss_function']:<40} {res['alpha']:<10} {res['train_size']:<10} {res['accuracy']:.2f}%")

if __name__ == "__main__":
    main()
