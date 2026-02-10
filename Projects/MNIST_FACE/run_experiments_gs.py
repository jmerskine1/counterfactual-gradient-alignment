import os
import subprocess
import re
import yaml
import itertools
import json
import time
from datetime import datetime

# GS Experiment Configuration for MNIST_FACE
PATH_TYPE = 'raw'
LOSS_FUNCTIONS = ['multiclass_split_gs_embedding']
ALPHAS = [0.3, 0.5] 
TRAIN_SIZES = [10, 20, 50, 100, 200]

CONFIG_PATH = 'Projects/MNIST_FACE/config.yaml'
RESULTS_FILE = 'Projects/MNIST_FACE/experiment_results_gs.json'
MANIFEST_FILE = 'Projects/MNIST_FACE/experiment_manifest_gs.txt'

def update_config(loss_fn, alpha, train_size):
    with open(CONFIG_PATH, 'r') as f:
        config_text = f.read()

    # Update dataset path
    dataset_name = f"integer_len200__connected_True__path_{PATH_TYPE}__split__boundary_False"
    config_text = re.sub(r'dataset: .*', f'dataset: {dataset_name}', config_text)
    
    config_text = re.sub(r'loss_function: .*', f'loss_function: {loss_fn}', config_text)
    config_text = re.sub(r'loss_mix: .*', f'loss_mix: {alpha}', config_text)
    config_text = re.sub(r'init_samplesize: .*', f'init_samplesize: {train_size}', config_text)
    config_text = re.sub(r'active_sampling: .*', 'active_sampling: True', config_text)
    config_text = re.sub(r'visualise: .*', 'visualise: false', config_text)
    config_text = re.sub(r'visualise_embeddings: .*', 'visualise_embeddings: false', config_text)
    config_text = re.sub(r'epochs: .*', 'epochs: 5', config_text)

    with open(CONFIG_PATH, 'w') as f:
        f.write(config_text)

def run_pipeline():
    try:
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
            print(line, end='') 
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
    
    all_experiments = list(itertools.product(LOSS_FUNCTIONS, ALPHAS, TRAIN_SIZES))
    
    print(f"Starting {len(all_experiments)} MNIST_FACE GS experiments...")
    
    with open(MANIFEST_FILE, 'w') as f:
        f.write("Timestamp, Loss Function, Alpha, Train Size, Accuracy, Output File\n")

    for loss_fn, alpha, train_size in all_experiments:
        print(f"\n" + "="*50)
        print(f"Running GS: Loss={loss_fn}, Alpha={alpha}, Size={train_size}")
        print("="*50)
        
        update_config(loss_fn, alpha, train_size)
        
        start_time = time.time()
        accuracy, output_file = run_pipeline()
        duration = time.time() - start_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if accuracy is not None:
            print(f"\n  -> RESULT Accuracy: {accuracy}% (Time: {duration:.2f}s)")
            results.append({
                'timestamp': timestamp,
                'loss_function': loss_fn,
                'alpha': alpha,
                'train_size': train_size,
                'accuracy': accuracy,
                'duration': duration,
                'output_file': output_file
            })
            with open(MANIFEST_FILE, 'a') as f:
                f.write(f"{timestamp}, {loss_fn}, {alpha}, {train_size}, {accuracy}, {output_file}\n")
        else:
            print("\n  -> FAILED run.")
            
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
