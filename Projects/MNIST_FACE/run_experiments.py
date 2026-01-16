import os
import subprocess
import re
import yaml
import itertools
import json
import time
from datetime import datetime

# Experiment Configuration for MNIST_FACE (No Active Learning)
PATH_TYPE = 'raw'
LOSS_FUNCTIONS = [
    'multiclass_cross_entropy',
    'multiclass_split_loss_embedding_relu',
    'multiclass_split_loss_embedding_softplus',
    'multiclass_split_loss_embedding_sign'
]

ALPHAS = [0.3, 0.5] 
TRAIN_SIZES = [10, 50, 200]

CONFIG_PATH = 'Projects/MNIST_FACE/config.yaml'
RESULTS_FILE = 'Projects/MNIST_FACE/experiment_results_no_active.json'
MANIFEST_FILE = 'Projects/MNIST_FACE/experiment_manifest_no_active.txt'

def update_config(loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()

    # Update dataset path
    dataset_name = f"integer_len200__connected_True__path_{PATH_TYPE}__split__boundary_False"
    config_text = re.sub(r'dataset: .*', f'dataset: {dataset_name}', config_text)
    
    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'init_samplesize: .*', f'init_samplesize: {train_size}', config_text)
    config_text = re.sub(r'active_sampling: .*', 'active_sampling: False', config_text)
    config_text = re.sub(r'visualise: .*', 'visualise: false', config_text)
    config_text = re.sub(r'visualise_embeddings: .*', 'visualise_embeddings: false', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 5', config_text)

    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

def run_pipeline():
    try:
        # Run pipeline and stream output to avoid timeout
        process = subprocess.Popen(
            ['python3', 'Projects/MNIST_FACE/pipeline.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        full_output = []
        for line in process.stdout:
            print(line, end='') # Stream to parent stdout
            full_output.append(line)
        
        process.wait()
        output = "".join(full_output)
        
        if process.returncode != 0:
            print(f"Pipeline failed with return code {process.returncode}")
            return None, "Failed"

        matches_acc = re.findall(r'Validation | Loss: [\d\.]+, Accuracy: ([\d\.]+)%', output)
        accuracy = float(matches_acc[-1]) if matches_acc else None
        
        matches_file = re.findall(r'Loading and saving to :  (MODEL_ENSEMBLE_.*)', output)
        output_file = matches_file[-1].strip() if matches_file else "Unknown"
        
        return accuracy, output_file
            
    except Exception as e:
        print(f"Error running pipeline: {e}")
        return None, "Error"

def main():
    results = []
    
    ce_experiments = list(itertools.product(['multiclass_cross_entropy'], [0.0], TRAIN_SIZES))
    dir_experiments = list(itertools.product(
        [l for l in LOSS_FUNCTIONS if l != 'multiclass_cross_entropy'],
        ALPHAS,
        TRAIN_SIZES
    ))
    
    all_experiments = ce_experiments + dir_experiments
    
    print(f"Starting {len(all_experiments)} MNIST_FACE (No AL) experiments...")
    
    with open(MANIFEST_FILE, 'w') as f:
        f.write("Timestamp, Path Type, Loss Function, Alpha, Train Size, Accuracy, Output File\n")

    for loss_fn, alpha, train_size in all_experiments:
        print(f"\n" + "="*50)
        print(f"Running: Loss={loss_fn}, Alpha={alpha}, Size={train_size}")
        print("="*50)
        
        update_config(loss_fn, alpha, train_size)
        
        start_time = time.time()
        accuracy, output_file = run_pipeline()
        duration = time.time() - start_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if accuracy is not None:
            print(f"\n  -> RESULT Accuracy: {accuracy}% (Time: {duration:.2f}s)")
            
            result_entry = {
                'timestamp': timestamp,
                'path_type': PATH_TYPE,
                'loss_function': loss_fn,
                'alpha': alpha,
                'train_size': train_size,
                'accuracy': accuracy,
                'duration': duration,
                'output_file': output_file
            }
            results.append(result_entry)
            
            with open(MANIFEST_FILE, 'a') as f:
                f.write(f"{timestamp}, {PATH_TYPE}, {loss_fn}, {alpha}, {train_size}, {accuracy}, {output_file}\n")
        else:
            print("\n  -> FAILED run.")
            
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

    print("\n\n=== MNIST_FACE (No AL) Experiment Summary ===")
    print(f"{ 'Loss Function':<40} {'Alpha':<10} {'Size':<10} {'Accuracy':<10}")
    print("-" * 75)
    for res in sorted(results, key=lambda x: x['accuracy'], reverse=True):
        print(f"{res['loss_function']:<40} {res['alpha']:<10} {res['train_size']:<10} {res['accuracy']:.2f}%")

if __name__ == "__main__":
    main()