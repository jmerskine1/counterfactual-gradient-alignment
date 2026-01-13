# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]

## Standard libraries
import numpy as np
import pickle

## Progress bar
from tqdm.auto import tqdm

#ML Libraries
import jax
import torch
import optax

import torch.utils.data as data

# Custom Libraries
from gradient_supervision_package.library.custom_models   import SimpleClassifier, MLP, CNN,  GSPaper, GSPaperNew, GSPaper2, GSPaper3, BagOfWordsClassifier, BagOfWordsClassifierSimple, BagOfWordsClassifierSingle, TextClassifierEmbeddingsSetfit
from gradient_supervision_package.library.custom_datasets import customDataset
from gradient_supervision_package.library.loss_functions import loss_functions
from gradient_supervision_package.library.knowledge_functions import knowledge_functions
from gradient_supervision_package.library.utilities import (visualise_classes, numpy_collate, custom_collate,
                        reduce_dataset, compute_metrics, generate_results, generate_results_ensemble, create_train_state, boundary_filter, save_stats, train_one_epoch, combine_datasets)



"""
Load DATASET
"""
data_path = 'data/'
data_name = 'setfit_ft_untrained'


with open(data_path+data_name+'.pkl', 'rb') as file:
    datasets = pickle.load(file)

output_path = "/Users/jonathanerskine/University of Bristol/gradient_supervision/ecai_25/model_outputs/try_GS/"

"""
Initialise Parameters
"""
seed = 42

n_models = 2

n_vectors = len(datasets['train']['original']['X']['vector'][0])

rng = jax.random.PRNGKey(seed)

rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)

"""
Model Parameters
"""
loss_name = "cross_entropy" # "direction", 'cross_entropy_batch', 'cross_entropy_l2', 'direction', 'direction_interactive' & more - see loss functions
overwrite = True
learning_rate = 0.01
batch_size = 64

# Define a learning rate schedule (e.g., exponential decay)
learning_rate_schedule = optax.exponential_decay(
    init_value=1.0,  # Starting learning rate
    transition_steps=20,  # Steps after which decay happens
    decay_rate=0.5,  # Decay factor
    transition_begin=0,  # When to start the decay
    staircase=False  # Set to True for a staircase effect
)

sgd_opt = optax.sgd(learning_rate=0.01,momentum=0.8 )
adam_opt = optax.adam(learning_rate=learning_rate)
scheduled_adadelta = optax.adadelta(learning_rate=learning_rate_schedule, weight_decay=0.05)
scheduled_adam = optax.adam(learning_rate=learning_rate_schedule)#, weight_decay=0.05)
optimiser = adam_opt


warmup_steps = 3

final_lr = 1e-3

schedule = optax.join_schedules(
    schedules=[
        optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps),
        optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=0.9)
    ],
    boundaries=[warmup_steps]
)

# optimiser = optax.adam(schedule)

chained_optax = optax.chain(
    optax.clip_by_global_norm(1.0),  # Clip gradients
    optax.adam(schedule)
)
optimiser = chained_optax

# optimiser = scheduled_adam

optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]

# model = SimpleClassifier(8,1)
# model = MLP(64,1)
ensemble = {
            # 'models':[BagOfWordsClassifier(20000,50)]*n_models,
            'models':[SimpleClassifier(8,1)]*n_models,
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

for i in range(n_models):
    # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
    trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,vector_length=n_vectors)
    ensemble['train_states'].append(trained_state)
    ensemble['models'][i] = model
    ensemble['outputs']['params'][i] = trained_state.params
    





"""
Load embeddings and convert into pipeline datasets
"""


train_original = datasets['train']['original']
train_combined = combine_datasets(datasets['train']['original'],datasets['train']['modified'])

"""
SET FILTERING CRITERIA
"""
filtered = True

if filtered:
    percentage = 20
    training_data = customDataset(reduce_dataset(train_combined,percentage*0.01))
    filter_info = f"{percentage}percent_combined"
else:
    training_data = customDataset(train_combined)
    
training_dataloader = data.DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True, generator=torch.Generator().manual_seed(seed))

validation = customDataset(combine_datasets(datasets['dev']['original'],datasets['dev']['modified']))
validation_data_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_original = customDataset(datasets['test']['original'])
test_original_data_loader = data.DataLoader(test_original, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

test_modified = customDataset(datasets['test']['modified'])
test_modified_data_loader = data.DataLoader(test_modified, batch_size=batch_size, shuffle=True, collate_fn=custom_collate, drop_last=True)

output_name = f'MODEL_ENSEMBLE_{model_name}__OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{batch_size}__DATA_{data_name}_filtered_{filter_info}__LOSS_{loss_name}'
print("Loading and saving to : ", output_name)


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
for i in range(len(training_data)):
    if np.all(training_data.K['vector'][i]) == 0:
        filter_indices.append(i)

filter_indices = sorted(filter_indices, reverse=True)

# for i in filter_indices:
#     train_combined.drop(i)

print(f"Train dataset reduced to {len(training_data)} samples.")
last_val_acc = 0
for epoch in tqdm(range(50)):
            
    for m in range(n_models):
        model = ensemble['models'][m]
        trained_state = ensemble['train_states'][m]
        rng = ensemble['rngs'][m]
        
        trained_state, train_metrics = train_one_epoch(trained_state, training_dataloader,  
                                                        # model, loss_functions['direction_interactive_vectorized'])
                                                    #  model, loss_functions['direction_interactive'])
                                                    # model, loss_functions['direction_interactive2'])
                                                    #  model, loss_functions['gradient_supervision'],rng)
                                                    #  model, loss_functions['direction'],rng)
                                                    model, loss_functions[loss_name],rng)
        
        ensemble['outputs']['params'][m]=trained_state.params
        ensemble['train_states'][m] = trained_state
        
        

                                                    #  model, loss_functions['cross_entropy_l2'])
                                            
        # trained_state,metrics = train_one_epoch(trained_state, train_data_loader)
        # print(f"Epoch Loss: {train_metrics['loss']}, Epoch Accuracy: {train_metrics['accuracy'] * 100}")
    
    models = ensemble['models']
    ensemble_params = ensemble['outputs']['params']
    
    train_metrics = generate_results_ensemble(training_data,models,ensemble_params,name="Train")
    val_metrics = generate_results_ensemble(validation,models,ensemble_params,name="Validation")
    test_o_metrics = generate_results_ensemble(test_original,models,ensemble_params,name="Test Original")
    test_m_metrics = generate_results_ensemble(test_modified,models,ensemble_params,name="Test Modified")


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
print(f'Model saved to {output_path} as:\n\n{output_name}')
with open(output_path + output_name + '.pkl', 'wb') as file:
    pickle.dump(ensemble['outputs'],file)
