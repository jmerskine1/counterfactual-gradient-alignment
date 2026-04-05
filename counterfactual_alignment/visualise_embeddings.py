# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]
## Standard libraries
import numpy as np
import pickle
import yaml
import pickle as pkl
import os
import sys
from pathlib import Path

#ML Libraries
import jax

# Custom Libraries
from counterfactual_alignment.custom_datasets import customDataset
from counterfactual_alignment.utilities import combine_datasets, visualise_embeddings

p = sys.argv[1]
project_dir = Path(p).resolve().parents[2]

split_p = p.split('/')

print(f"Reading model {split_p[-1]} from dataset {split_p[-2]} from Project: {project_dir}")
config_path = os.path.join(project_dir,'config.yaml')

with open(config_path,'r') as file:
    config = yaml.unsafe_load(file)

"""Load Model"""

# model = "model_outputs/integer_len32_SIZE_250/MODEL_ENSEMBLE_6_BagOfWordsClassifierMultiClass__emb64_OPTIM_adabelief__LR_0.001__BATCHSIZE_32__trainsize_250__active_False_smplsize_5_div0.8_LOSS_multiclass_cross_entropy_mix_0.3.pkl"
# /Users/jonathanerskine/University of Bristol/gradient_supervision/counterfactual-gradient-alignment-cli/Projects/AGNEWS/model_outputs/integer_len32_SIZE_250/MODEL_ENSEMBLE_6_BagOfWordsClassifierMultiClass__emb64_OPTIM_adabelief__LR_0.001__BATCHSIZE_32__trainsize_250__active_False_smplsize_5_div0.8_LOSS_multiclass_cross_entropy_mix_0.3.pkl
with open(p,'rb') as f:
    model_data = pkl.load(f)

"""
Load DATASET
"""
# Build path to "data" folder inside that directory
data_folder = os.path.join(project_dir, 'data')

data_name = config['data_params']['dataset']

print(f"Loading dataset: \n{data_name} from \n{data_folder}\n")

with open(os.path.join(data_folder,data_name)+'.pkl', 'rb') as file:
    datasets = pickle.load(file)


n_classes = datasets['n_classes']


n_vectors = len(datasets['train']['original']['X'][0])
train_size = len(datasets['train']['original']['Y'])

rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)



"""
Load embeddings and convert into pipeline datasets
"""

if config['data_params']['control']:
    full_training_set = customDataset(combine_datasets(datasets['control']['original'], datasets['control']['original']))
else: 
    full_training_set = customDataset(datasets['train']['original'])
    # Filter out any case where counterfactual is the same as the original
if False:
    filter_indices = []
    for i in range(len(full_training_set)):
        if np.all(training_set.K['vector'][i]) == 0:
            filter_indices.append(i)

    filter_indices = sorted(filter_indices, reverse=True)

    for i in filter_indices:
        training_set.drop(i)
    
    print(f"Train dataset reduced to {len(training_set)} samples.")

training_set = full_training_set
validation = customDataset(datasets['dev']['original'])
test_original = customDataset(datasets['test']['original'])

datasets = {
    "Train": training_set,
    'Validation': validation,
    # 'Test': test_original
}





visualise_embeddings(datasets,model_data)