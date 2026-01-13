import numpy as np
import pickle
import yaml
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
import os

# Custom Libraries
from counterfactual_alignment.custom_datasets import datasets as custom_2d_datasets
from counterfactual_alignment.custom_datasets import customDataset
from counterfactual_alignment.knowledge_functions import knowledge_functions

# Get the directory where THIS script is located
project_dir = os.path.dirname(os.path.abspath(__file__))

# Load config to get dataset and knowledge function parameters
with open(os.path.join(project_dir, 'config.yaml'), 'r') as file:
    config = yaml.unsafe_load(file)

dataset_name = config['data_params']['dataset']
knowledge_func_name = config['data_params']['knowledge_func']
n_vec = config['data_params']['n_vec']
seed = config['hyperparams']['seed']
train_size = config['data_params']['train_size']
validation_size = config['data_params']['validation_size']

# --- Data Generation Parameters ---
# Total samples to generate for the full dataset before splitting
total_samples = train_size + validation_size + 200 # Add some for test and control sets
test_samples = 100
control_samples = 100 # Can be adjusted or removed if not needed

# Instantiate the dataset generator
rng = np.random.RandomState(seed)
full_dataset_generator = custom_2d_datasets[dataset_name](rng, total_samples)
X_full = full_dataset_generator.X
y_full = full_dataset_generator.Y

# --- Splitting Data ---
# Split into training, validation, and a pool for test/control
X_train_val, X_test_control, y_train_val, y_test_control = train_test_split(
    X_full, y_full, test_size=(test_samples + control_samples) / total_samples, random_state=seed, stratify=y_full
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=validation_size / (train_size + validation_size), random_state=seed, stratify=y_train_val
)

X_test, X_control, y_test, y_control = train_test_split(
    X_test_control, y_test_control, test_size=control_samples / (test_samples + control_samples), random_state=seed, stratify=y_test_control
)


# --- Generate Knowledge (K) for each split ---
knowledge_function = knowledge_functions[knowledge_func_name]

print("Generating K for training data...")
K_train = knowledge_function(X_train, full_dataset_generator.optimum_classifier, n_vec=n_vec)

print("Generating K for validation data...")
# For validation, dev, test, and control, we might not need "paths" or full counterfactuals,
# often just the original X, Y, and placeholder K are sufficient for evaluation.
# However, to maintain the structure, we can generate a simplified K or use a dummy.
# For now, let's generate full K for consistency if the function allows.
K_val = knowledge_function(X_val, full_dataset_generator.optimum_classifier, n_vec=n_vec)

print("Generating K for test data...")
K_test = knowledge_function(X_test, full_dataset_generator.optimum_classifier, n_vec=n_vec)

print("Generating K for control data...")
K_control = knowledge_function(X_control, full_dataset_generator.optimum_classifier, n_vec=n_vec)


# --- Assemble the final datasets dictionary ---
datasets_dict = {
    'train': {
        'original': {
            'X': X_train,
            'Y': y_train,
            'K': K_train
        }
    },
    'dev': { # Corresponds to validation
        'original': {
            'X': X_val,
            'Y': y_val,
            'K': K_val
        }
    },
    'test': {
        'original': {
            'X': X_test,
            'Y': y_test,
            'K': K_test
        }
    },
    'control': {
        'original': {
            'X': X_control,
            'Y': y_control,
            'K': K_control
        }
    },
    'n_classes': len(np.unique(y_full))
}

# --- Save the datasets dictionary ---
data_folder = os.path.join(project_dir, 'data')
os.makedirs(data_folder, exist_ok=True)

# Construct the filename using config parameters
# This filename should match what the pipeline expects, e.g., 'XSQUARED.pkl'
output_filename = f"{dataset_name}.pkl"
output_path = os.path.join(data_folder, output_filename)

with open(output_path, 'wb') as f:
    pickle.dump(datasets_dict, f)

print(f"Generated and saved dataset to {output_path}")