# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]

## Standard libraries
import numpy as np
import pickle
import yaml

## Progress bar
from tqdm.auto import tqdm

#ML Libraries
import jax
import optax

import os

import torch.utils.data as data

# Custom Libraries
from counterfactual_alignment.custom_models   import SimpleClassifier, MLP, CNN,  GSPaper, GSPaperNew, GSPaper2, GSPaper3, BagOfWordsClassifier, BagOfWordsClassifierSimple, BagOfWordsClassifierSingle, SentimentModel
from counterfactual_alignment.custom_datasets import customDataset
from counterfactual_alignment.loss_functions import loss_functions
from counterfactual_alignment.knowledge_functions import knowledge_functions
from counterfactual_alignment.utilities import (visualise_classes, numpy_collate, imdb_collate,
                        reduce_dataset, compute_metrics, generate_results, generate_results_ensemble, create_train_state, boundary_filter, save_stats, train_one_epoch, combine_datasets)



# Get the directory where THIS script is located
project_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(project_dir,'config.yaml'),'r') as file:
    config = yaml.unsafe_load(file)


"""
Load DATASET
"""
# Build path to "data" folder inside that directory
data_folder = os.path.join(project_dir, 'data')

data_name = config['data_params']['dataset']


with open(os.path.join(data_folder,data_name)+'.pkl', 'rb') as file:
    datasets = pickle.load(file)



output_path = project_dir + "/model_outputs/" + data_name + "/"
os.makedirs(output_path, exist_ok=True)

"""
Initialise Parameters
"""


n_models = 1
n_epochs = config['hyperparams']['epochs']
overwrite = True

n_vectors = len(datasets['train']['original']['X'][0])

rng = jax.random.PRNGKey(42)

rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)

"""
Model Parameters
"""
loss_name = config['hyperparams']['loss_function'] # "direction", 'cross_entropy_batch', 'cross_entropy_l2', 'direction', 'direction_interactive' & more - see loss functions
learning_rate = config['hyperparams']['learning_rate']
batch_size = config['hyperparams']['batch_size']


warmup_steps = 100
peak_lr = 1.0
final_lr = 1e-3

schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(init_value=0.0, end_value=peak_lr, transition_steps=warmup_steps),
        optax.exponential_decay(init_value=peak_lr, transition_steps=100, decay_rate=0.9)
    ],
    boundaries=[warmup_steps]
)

# optimiser = optax.adam(schedule)

optimiser = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(schedule)
)


# Define a learning rate schedule (e.g., exponential decay)
learning_rate_schedule = optax.exponential_decay(
    init_value=1.0,  # Starting learning rate
    transition_steps=50, 
    decay_rate=0.9,  # Decay factor
    transition_begin=2,  # When to start the decay
    staircase=False  # Set to True for a staircase effect
)

sgd_opt = optax.sgd(learning_rate=0.01,momentum=0.8 )
adam_opt = optax.adam(learning_rate=learning_rate)
adamw = optax.adamw(
    learning_rate=1e-3,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    weight_decay=0.0
)
adadelta = optax.adadelta(
    learning_rate=1,    # default
    rho=0.95,              # decay rate
    eps=1e-6
)
scheduled_adadelta = optax.adadelta(learning_rate=learning_rate_schedule, weight_decay=0.05)

optimiser = adamw
# optimiser = adam_opt
# optimiser = scheduled_adadelta

optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]

# model = SimpleClassifier(8,1)
# model = MLP(64,1)
ensemble = {
            # 'models':[BagOfWordsClassifier(20000,50)]*n_models,
            # 'models': [SentimentModel(20000,50)]*n_models,
            'models':[BagOfWordsClassifierSimple(20002,50)]*n_models,
            'rngs':jax.random.split(rng,n_models),
            'init_rngs':jax.random.split(init_rng,n_models),
            'train_states':[],
            'outputs':{
              'params':[None]*n_models,
              'results':{
                'Train':{'losses':[],'accuracy':[]},
                'Validation':{'losses':[],'accuracy':[]}, 
                'Test Original':{'losses':[],'accuracy':[]},
                'Test Modified':{'losses':[],'accuracy':[]}}
                }}


# model = GSPaper3(8,1)
model_name = type(ensemble['models'][0]).__name__


output_name = f'MODEL_ENSEMBLE_{n_models}_{model_name}__OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{batch_size}__DATA_{data_name}_filtered__LOSS_{loss_name}'
print("Loading and saving to : ", output_name)

for i in range(n_models):
    trained_state, model = create_train_state(ensemble['models'][i],optimiser,vector_length=n_vectors, key=ensemble['init_rngs'][i])
    # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
    ensemble['train_states'].append(trained_state)
    ensemble['models'][i] = model
    ensemble['outputs']['params'][i] = trained_state.params

"""
Load embeddings and convert into pipeline datasets
"""



train_original = customDataset(datasets['train']['original'])
train_original_data_loader = data.DataLoader(train_original, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)

train_combined = customDataset(combine_datasets(datasets['train']['original'],datasets['train']['modified']))
train_combined_data_loader = data.DataLoader(train_combined, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=False)

validation = customDataset(combine_datasets(datasets['dev']['original'],datasets['dev']['modified']))
validation_data_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)

test_original = customDataset(datasets['test']['original'])
test_original_data_loader = data.DataLoader(test_original, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)

test_modified = customDataset(datasets['test']['modified'])
test_modified_data_loader = data.DataLoader(test_modified, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)


if not overwrite:
    try:

        with open(output_path + output_name+'.pkl', 'rb') as file: ## remove this line to load model
            res = pickle.load(file)

        ensemble['outputs']['params'] = res['params']

        for i,trainstate in enumerate(ensemble['train_states']):
            ensemble['train_states'][i] = ensemble['train_states'][i].replace(params = res['params'][i])
        
        ensemble['outputs']['results'] = res['results']

        print(f'Model loaded from {output_name}')
        
    except:
        pass

# Filter out any case where counterfactual is the same as the original
filter_indices = []
for i in range(len(train_combined)):
    if np.all(train_combined.K['vector'][i]) == 0:
        filter_indices.append(i)

filter_indices = sorted(filter_indices, reverse=True)

# for i in filter_indices:
#     train_combined.drop(i)

print(f"Train dataset reduced to {len(train_combined)} samples.")
last_val_acc = 0
for epoch in tqdm(range(n_epochs)):
            
    for m in range(n_models):
        model = ensemble['models'][m]
        trained_state = ensemble['train_states'][m]
        rng = ensemble['rngs'][m]

        for batch in train_combined_data_loader:
            trained_state, train_metrics = train_one_epoch(trained_state, batch, model, loss_functions[loss_name],rng)
            
        
        ensemble['outputs']['params'][m]=trained_state.params
        ensemble['train_states'][m] = trained_state
        
        

                                                    #  model, loss_functions['cross_entropy_l2'])
                                            
        # trained_state,metrics = train_one_epoch(trained_state, train_data_loader)
        # print(f"Epoch Loss: {train_metrics['loss']}, Epoch Accuracy: {train_metrics['accuracy'] * 100}")
    
    models = ensemble['models']
    ensemble_params = ensemble['outputs']['params']
    
    train_metrics = generate_results_ensemble(train_combined.X,train_combined.Y,models,ensemble_params,name="Train")
    val_metrics = generate_results_ensemble(validation.X,validation.Y,models,ensemble_params,name="Validation")
    test_o_metrics = generate_results_ensemble(test_original.X,test_original.Y,models,ensemble_params,name="Test Original")
    test_m_metrics = generate_results_ensemble(test_modified.X,test_modified.Y,models,ensemble_params,name="Test Modified")


    ensemble['outputs']['results']['Train']['losses'].append(train_metrics['loss'])
    ensemble['outputs']['results']['Train']['accuracy'].append(train_metrics['accuracy'])
    
    ensemble['outputs']['results']['Validation']['losses'].append(val_metrics['loss'])
    ensemble['outputs']['results']['Validation']['accuracy'].append(val_metrics['accuracy'])
    
    ensemble['outputs']['results']['Test Original']['losses'].append(test_o_metrics['loss'])
    ensemble['outputs']['results']['Test Original']['accuracy'].append(test_o_metrics['accuracy'])

    ensemble['outputs']['results']['Test Modified']['losses'].append(test_m_metrics['loss'])
    ensemble['outputs']['results']['Test Modified']['accuracy'].append(test_m_metrics['accuracy'])

    # if val_metrics['accuracy']<last_val_acc:
    #     break
    # else:
    #     last_val_acc = val_metrics['accuracy']
    if epoch%5==0 or epoch == n_epochs-1:
        print(f"saving: {output_name}")
        with open(output_path + output_name + '.pkl', 'wb') as file:
            pickle.dump(ensemble['outputs'],file)
