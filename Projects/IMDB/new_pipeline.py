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
from functools import partial
import argparse

import torch.utils.data as data

# Custom Libraries
from counterfactual_alignment.custom_models   import SimpleClassifier, MLP, CNN,  GSPaper, GSPaperNew, GSPaper2, GSPaper3, BagOfWordsClassifier, BagOfWordsClassifierSimple, BagOfWordsClassifier2Layer, BagOfWordsClassifierSingle, SentimentModel
from counterfactual_alignment.custom_models   import custom_models
from counterfactual_alignment.custom_datasets import customDataset
from counterfactual_alignment.loss_functions import loss_functions
from counterfactual_alignment.knowledge_functions import knowledge_functions
from counterfactual_alignment.utilities import (visualise_classes, imdb_collate, select_informative_samples,
                        reduce_dataset, compute_metrics, generate_results, generate_results_ensemble, create_train_state, 
                        boundary_filter, save_stats, train_one_epoch, combine_datasets,reinit_layer)

# Get the directory where THIS script is located
project_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-oc", default='config.yaml', help="Config file name")
args = parser.parse_args()

config_path = os.path.join(project_dir,args.config)


with open(config_path,'r') as file:
    config = yaml.unsafe_load(file)

active_sampling = config['data_params']['active_sampling']
sample_size = config['data_params']['sample_size']
"""
Load DATASET
"""
# Build path to "data" folder inside that directory
data_folder = os.path.join(project_dir, 'data')

data_name = config['data_params']['dataset']


with open(os.path.join(data_folder,data_name)+'.pkl', 'rb') as file:
    datasets = pickle.load(file)

try:
    n_classes = datasets['n_classes']
except:
    n_classes = len(np.unique(datasets['train']['original']['Y']))

output_path = project_dir + "/model_outputs/" + data_name + "/"
os.makedirs(output_path, exist_ok=True)

"""
Initialise Parameters
"""


n_models = config['hyperparams']['n_models']
n_epochs = config['hyperparams']['epochs']
overwrite = True


rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)

"""
Model Parameters
"""
loss_function = partial(loss_functions[config['hyperparams']['loss_function']], alpha=config['hyperparams']['loss_mix'])

learning_rate = config['hyperparams']['learning_rate']
batch_size = config['hyperparams']['batch_size']

warmup_steps = 100
peak_lr = 1.0
final_lr = 1e-3

weight_decay = 1.0e-4

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
    init_value=learning_rate,  # Starting learning rate
    transition_steps=50, 
    decay_rate=0.9,  # Decay factor
    transition_begin=2,  # When to start the decay
    staircase=False  # Set to True for a staircase effect
)

sgd_opt = optax.sgd(learning_rate=0.01,momentum=0.8 )
adam_opt = optax.adam(learning_rate=learning_rate)
adamw = optax.adamw(
    learning_rate=learning_rate,
    b1=0.9,
    b2=0.95,
    eps=1e-8,
    weight_decay=weight_decay,
    nesterov=False
)
adadelta = optax.adadelta(
    learning_rate=learning_rate,    # default
    rho=0.95,              # decay rate
    eps=1e-6
)
scheduled_adadelta = optax.adadelta(learning_rate=learning_rate_schedule, weight_decay=0.05)

optimiser = adamw

# optimiser = adadelta
# optimiser = adam_opt
# optimiser = sgd_opt
# optimiser = scheduled_adadelta

optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]

# model = SimpleClassifier(8,1)
# model = MLP(64,1)
embedding_size = config['hyperparams'].get('embedding_size',64)
ensemble = {
            # 'models':[BagOfWordsClassifier(20000,50)]*n_models,
            # 'models': [SentimentModel(20000,50)]*n_models,
            'models':[
                custom_models[config['hyperparams']['model']](
                    config['hyperparams']['vocabulary_size'],config['hyperparams']['embedding_size'])
                    ]*n_models,
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



"""
Load embeddings and convert into pipeline datasets
"""

control = False

if control:
    full_training_set = customDataset(datasets['control']['original'])
else: 
    # full_training_set = customDataset(datasets['train']['original'])
    full_training_set = customDataset(combine_datasets(datasets['train']['original'],datasets['train']['modified']))
    # full_training_set = customDataset(reduce_dataset(datasets['train']['original'],0.1))
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

n_vectors = len(full_training_set.X[0])
train_size = len(full_training_set.Y)



if active_sampling:
    init_samplesize = 600
    nprng = np.random.default_rng(123)
    subsample_indices = nprng.choice(train_size,init_samplesize,replace=False)
    unsampled = set(np.arange(0,train_size)) - set(subsample_indices)
    training_set = full_training_set.subset(subsample_indices)
else:
    training_set = full_training_set

print(f"Training on {len(training_set)} samples.")


train_original_data_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=False)


validation = customDataset(datasets['dev']['original'])
validation_data_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)

test_original = customDataset(datasets['test']['original'])
test_original_data_loader = data.DataLoader(test_original, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)

test_modified = customDataset(datasets['test']['modified'])
test_modified_data_loader = data.DataLoader(test_modified , batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)


output_name = (f"MODEL_ENSEMBLE_{n_models}_{model_name}__"
               f"emb{config['hyperparams']['embedding_size']}_"
               f"OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{batch_size}__"
               f"trainsize_{train_size}__active_{active_sampling}_smplsize_{sample_size}_div{config['data_params']['diversity']}_"
               f"LOSS_{config['hyperparams']['loss_function']}_mix_{config['hyperparams']['loss_mix']}")

print("Loading and saving to : ", output_name)

for i in range(n_models):
    trained_state, model = create_train_state(ensemble['models'][i],optimiser,vector_length=n_vectors, key=ensemble['init_rngs'][i])
    # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
    ensemble['train_states'].append(trained_state)
    ensemble['models'][i] = model
    ensemble['outputs']['params'][i] = trained_state.params


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


last_val_acc = 0
for epoch in tqdm(range(n_epochs)):
        
            
    for m in range(n_models):
        
        model = ensemble['models'][m]
        trained_state = ensemble['train_states'][m]
        rng = ensemble['rngs'][m]
        
        for batch in train_original_data_loader:
            
            
            trained_state, train_metrics = train_one_epoch(trained_state, batch, model, loss_function, rng) #loss_functions[config["hyperparams"]["loss_function"]],rng)
            
        
        ensemble['outputs']['params'][m]=trained_state.params
        ensemble['train_states'][m] = trained_state
        
        

                                                    #  model, loss_functions['cross_entropy_l2'])
                                            
        # trained_state,metrics = train_one_epoch(trained_state, train_data_loader)
        # print(f"Epoch Loss: {train_metrics['loss']}, Epoch Accuracy: {train_metrics['accuracy'] * 100}")
    
    models = ensemble['models']
    ensemble_params = ensemble['outputs']['params']
    
    train_metrics = generate_results_ensemble(training_set.X,training_set.Y,models,ensemble_params,name="Train")
    val_metrics = generate_results_ensemble(validation.X,validation.Y,models,ensemble_params,name="Validation")
    test_o_metrics = generate_results_ensemble(test_original.X,test_original.Y,models,ensemble_params,name="Test Original")
    test_m_metrics = generate_results_ensemble(test_modified.X,test_modified.Y,models,ensemble_params,name="Test Modified")


    ensemble['outputs']['results']['Train']['losses'].append(train_metrics['loss'])
    ensemble['outputs']['results']['Train']['accuracy'].append(train_metrics['accuracy'])
    
    ensemble['outputs']['results']['Validation']['losses'].append(val_metrics['loss'])
    ensemble['outputs']['results']['Validation']['accuracy'].append(val_metrics['accuracy'])
    
    ensemble['outputs']['results']['Test Original']['losses'].append(test_o_metrics['loss'])
    ensemble['outputs']['results']['Test Original']['accuracy'].append(test_o_metrics['accuracy'])

    # if val_metrics['accuracy']<last_val_acc:
    #     break
    # else:
    #     last_val_acc = val_metrics['accuracy']
    if epoch%5==0 or epoch == n_epochs-1:
        print(f"saving: {output_name}")
        with open(output_path + output_name + '.pkl', 'wb') as file:
            pickle.dump(ensemble['outputs'],file)

    if active_sampling and epoch%2==0 and epoch > 1  and len(training_set) < train_size:
        print("Active learning: selecting new samples...")
        print(len(unsampled))
        # Use the first model of the ensemble (or average predictions)
        model = ensemble['models'][0]
        params = ensemble['outputs']['params'][0]
        
        remainder = full_training_set.subset(list(unsampled))
        
        # remainder = imdb_collate([train_original[i] for i in unsampled])
        
        logits, embeds = model.apply(
            {'params': trained_state.params},
            remainder.X,
            train=False,
            rngs={'dropout': rng}
        )
        probs = jax.nn.sigmoid(logits)
        # print("NDIM:",probs.ndim)

        new_indices = select_informative_samples(probs, embeds, k=sample_size, diversity_weight=config['data_params']['diversity'])
        unsampled = unsampled - set(new_indices)
        
        new_subset = remainder.subset(new_indices)
        # new_subset = [remainder[i] for i in new_indices]
        # print(new_subset['X'])
        training_set = training_set.combine(new_subset)
        print(f"New training set size: {len(training_set)}")
        # new_subset = combine_datasets(initial_subset,new_subset)
        train_original_data_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=imdb_collate, drop_last=True)
        # Move selected samples from unlabeled to training
        # for m in range(n_models):
        #     ensemble['train_states'][m] = reinit_layer(ensemble['train_states'][m],
        #                                                ensemble['models'][m],
        #                                                'linear1',
        #                                                vector_length=n_vectors,
        #                                                embedding_dim=embedding_size,
        #                                                rng=ensemble['rngs'][m])
        if False:
            for i in range(n_models):
                adamw = optax.adamw(
                    learning_rate=learning_rate,
                    b1=0.9,
                    b2=0.95,
                    eps=1e-8,
                    weight_decay=weight_decay,
                    nesterov=False
                    )
                trained_state, model = create_train_state(ensemble['models'][i],adamw,vector_length=n_vectors, key=ensemble['init_rngs'][i])
                # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
                # optimiser.init()
                ensemble['train_states'][m]=trained_state
                ensemble['models'][i] = model
                ensemble['outputs']['params'][i] = trained_state.params