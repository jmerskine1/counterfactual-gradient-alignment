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
from gradient_supervision_package.library.custom_datasets import customDataset, genCustomDataset
from gradient_supervision_package.library.custom_datasets import datasets as custom_datasets
from gradient_supervision_package.library.loss_functions import loss_functions
from gradient_supervision_package.library.knowledge_functions import knowledge_functions
from gradient_supervision_package.library.utilities import (visualise_classes, numpy_collate, custom_collate, custom_collate_2D,
                        reduce_dataset, compute_metrics, generate_results, generate_results_ensemble, create_train_state, boundary_filter, save_stats, train_one_epoch, combine_datasets, plotEpoch)


import yaml

config_file = "/Users/jonathanerskine/University of Bristol/gradient_supervision/ecai_25/config.yaml"
with open(config_file,'r') as file:
    config = yaml.unsafe_load(file)[0]


"""
Model Parameters
"""
 # "direction", 'cross_entropy_batch', 'cross_entropy_l2', 'direction', 'direction_interactive' & more - see loss functions
overwrite = True

loss_name = config['hyperparams']['loss_function']
learning_rate = config['hyperparams']['learning_rate']
batch_size = config['hyperparams']['batch_size']

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

"""
Initialise Parameters
"""
output_path = "/Users/jonathanerskine/University of Bristol/gradient_supervision/ecai_25/model_outputs/try_GS/"

seed = 42

n_models = 2

rng = jax.random.PRNGKey(seed)

rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)


"""
Gen Datasets
"""

train = genCustomDataset(custom_datasets[config['data_params']['dataset']],config['data_params']['train_size'],knowledge_functions[config['data_params']['knowledge_func']],
                                                            train=True, 
                                                            visualise=config['visualisation']['visualise'],
                                                            seed=config['hyperparams']['seed'],
                                                            n_vec = config['data_params']['n_vec'])

training_dataloader = data.DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=custom_collate_2D, generator=torch.Generator().manual_seed(seed))

validation = genCustomDataset(custom_datasets[config['data_params']['dataset']],config['data_params']['validation_size'],knowledge_functions[config['data_params']['knowledge_func']],
                                                            train=False, visualise=False,seed=config['hyperparams']['seed'])
validation_data_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=custom_collate_2D)



# model = SimpleClassifier(8,1)
# model = MLP(64,1)

n_vectors = len(train.X['vector'][0])

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

data_name = config['data_params']['dataset']
output_name = f'MODEL_ENSEMBLE_{model_name}__OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{batch_size}__DATA_{data_name}__LOSS_{loss_name}_alpha_2__SIZE_{config['data_params']['train_size']}'

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
plot_states = []
last_val_acc = 0

n_epochs = 100

for epoch in tqdm(range(n_epochs)):
            
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
        if m == 0:
            plot_states.append(trained_state)
        
        

                                                    #  model, loss_functions['cross_entropy_l2'])
                                            
        # trained_state,metrics = train_one_epoch(trained_state, train_data_loader)
        # print(f"Epoch Loss: {train_metrics['loss']}, Epoch Accuracy: {train_metrics['accuracy'] * 100}")
    
    models = ensemble['models']
    ensemble_params = ensemble['outputs']['params']
    
    train_metrics = generate_results_ensemble(train.X,train.Y,models,ensemble_params,name="Train")
    val_metrics = generate_results_ensemble(validation.X,validation.Y,models,ensemble_params,name="Validation")
    


    ensemble['outputs']['results']['Train']['losses'].append(train_metrics['loss'])
    ensemble['outputs']['results']['Train']['accuracy'].append(train_metrics['accuracy'])
    
    ensemble['outputs']['results']['Validation']['losses'].append(val_metrics['loss'])
    ensemble['outputs']['results']['Validation']['accuracy'].append(val_metrics['accuracy'])
    

    # if val_metrics['accuracy']<last_val_acc:
    #     break
    # else:
    #     last_val_acc = val_metrics['accuracy']




total_epochs = len(ensemble['outputs']['results']['Train']['accuracy'])

if config['visualisation']['video']:
    plotEpoch(ensemble['init_rngs'][0],
          train.X['vector'],train.Y,
          ensemble['models'][0],
          plot_states,
          plot_type='video',
          name = output_name)

print(f'Model saved to {output_path} as:\n\n{output_name}. \nTrained for {n_epochs} epochs (Total: {total_epochs})')
with open(output_path + output_name + '.pkl', 'wb') as file:
    pickle.dump(ensemble['outputs'],file)
