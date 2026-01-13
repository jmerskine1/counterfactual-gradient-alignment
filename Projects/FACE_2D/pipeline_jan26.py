
# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]

## Standard libraries
import numpy as np
import pickle
import yaml
from functools import partial

import matplotlib.pyplot as plt

## Progress bar
from tqdm.auto import tqdm

#ML Libraries
import jax
import optax

import os

import torch.utils.data as data

# Custom Libraries
from counterfactual_alignment.custom_models   import SimpleClassifier, SimpleClassifier_v2, MLP, CNN,  GSPaper, GSPaperNew, GSPaper2, GSPaper3, BagOfWordsClassifier, BagOfWordsClassifierSimple, BagOfWordsClassifierSingle
from counterfactual_alignment.custom_datasets import customDataset, genCustomDataset
from counterfactual_alignment.custom_datasets import datasets as custom_datasets
from counterfactual_alignment.loss_functions import loss_functions
from counterfactual_alignment.knowledge_functions import knowledge_functions
from counterfactual_alignment.knowledge_functions import counterfactual_vector_paths, counterfactual_feasible_vector_paths, counterfactual_breadcrumbs
import  counterfactual_alignment.utilities  as ut

import torch


# Get the directory where THIS script is located

project_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(project_dir,'config.yaml'),'r') as file:
    config = yaml.unsafe_load(file)

# with open('config.yaml','r') as file:
#     config = yaml.unsafe_load(file)
"""
Generate DATASET
"""
data_name = config['data_params']['dataset']

train = genCustomDataset(custom_datasets[data_name],config['data_params']['train_size'],None,
                         train=True,
                         visualise=config['visualisation']['visualise'],
                         seed=config['hyperparams']['seed'],
                         n_vec = config['data_params']['n_vec'])


validation = genCustomDataset(custom_datasets[data_name],config['data_params']['validation_size'],None,
                         train=False,
                         visualise=config['visualisation']['visualise'],
                         seed=config['hyperparams']['seed'],
                         n_vec = config['data_params']['n_vec'])

KX = []
KY = []
KK = []

classes = list(np.unique(train.Y))
n_classes = len(classes)
n_points = len(train.X)

for class_i in classes:
    for point in [train.X[i] for i, y in enumerate(train.Y) if y == class_i]:
        
        
        other_classes = [classes[i] for i, x in enumerate(classes) if x != class_i]
        
        KX_i = []
        KY_i = []
        KK_i = []

        for other_class in other_classes:
            
            idx = np.random.choice([i for i, x in enumerate(train.Y) if x == other_class])
            
            breadcrumbs  = counterfactual_breadcrumbs(train.X[train.X==point],
                                                        train.X[idx],
                                                        classifier = train.data.optimum_classifier,
                                                        n_breadcrumbs=3)
            
            
            i = 0
            print(breadcrumbs['labels'])
            print(breadcrumbs)
            selected_class = breadcrumbs['labels'][i]

            KX_i.append(breadcrumbs['origins'][i])
            
            KY_i.append(class_i)
            KK_i.append(breadcrumbs['origins'][i] + breadcrumbs['vectors'][i])
            
            while selected_class == class_i:
                print(selected_class,class_i)
                i += 1

                selected_class = breadcrumbs['labels'][i]
                
                KX_i.append(breadcrumbs['origins'][i])
                KY_i.append(class_i)
                KK_i.append(breadcrumbs['origins'][i] + breadcrumbs['vectors'][i])
                
        KX.append(np.array(KX_i))
        KY.append(np.array(KY_i))
        KK.append(np.array(KK_i))
        
train.K = {'X': KX,
           'Y': KY,
           'K': KK}

output_path = project_dir + "/model_outputs/" + data_name + "/"
# output_path = "outputs/" + data_name + "/"
os.makedirs(output_path, exist_ok=True)

"""
Initialise Parameters
"""

n_models = 6
n_epochs = config['hyperparams']['epochs']
overwrite = True

n_vectors = len(train.X[0])

rng = jax.random.PRNGKey(42)

rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)

loss_function = partial(loss_functions[config['hyperparams']['loss_function']], alpha=config['hyperparams']['loss_mix'])

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
    init_value=1,  # Starting learning rate
    transition_steps=50,  
    decay_rate=0.9,  # Decay factor
    transition_begin=10,  # When to start the decay
    staircase=False  # Set to True for a staircase effect
)

sgd_opt = optax.sgd(learning_rate=0.01,momentum=0.8 )
adam_opt = optax.adam(learning_rate=learning_rate)
scheduled_adadelta = optax.adadelta(learning_rate=learning_rate_schedule, weight_decay=0.05)
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

# optimiser = adamw
# optimiser = scheduled_adadelta
# optimiser = adadelta
optimiser = adam_opt

optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]



ensemble = {
            # 'models':[BagOfWordsClassifier(20000,50)]*n_models,
            'models':[SimpleClassifier_v2(*config['hyperparams']['model_io'])]*n_models,
            'rngs':jax.random.split(rng,n_models),
            'init_rngs':jax.random.split(init_rng,n_models),
            'train_states':[],
            'outputs':{
              'params':[None]*n_models,
              'results':{
                'Train':{'losses':[],'accuracy':[]},
                'Validation':{'losses':[],'accuracy':[]}, 
                }
                }}


# model = GSPaper3(8,1)
model_name = type(ensemble['models'][0]).__name__
output_name = f"MODEL_ENSEMBLE_{n_models}_{model_name}__OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{config['hyperparams']['batch_size']}__LOSS_{loss_name}_alpha_{config['hyperparams']['loss_mix']}__SIZE{config['data_params']['train_size']}"
print("Loading and saving to : ", output_name)

for i in range(n_models):
    # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
    trained_state = ut.create_train_state(ensemble['models'][i],optimiser,vector_length=n_vectors,key = ensemble['init_rngs'][i])
    ensemble['train_states'].append(trained_state)
    
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

# Filter out any case where counterfactual is the same as the original
plot_states = []
last_val_acc = 0



for epoch in tqdm(range(n_epochs)):
            
    for m in range(n_models):
        model = ensemble['models'][m]
        trained_state = ensemble['train_states'][m]
        rng = ensemble['rngs'][m]
        batch = {
            "X":train.X,
            "Y":train.Y,
            "K":train.K.copy()
        }
        
        trained_state, train_metrics = ut.train_one_epoch(trained_state, batch, model, loss_function,rng)
            
        
        ensemble['outputs']['params'][m]=trained_state.params
        ensemble['train_states'][m] = trained_state
        if m == 0:
            plot_states.append(trained_state)
        
        

                                                    #  model, loss_functions['cross_entropy_l2'])
                                            
        # trained_state,metrics = train_one_epoch(trained_state, train_data_loader)
        # print(f"Epoch Loss: {train_metrics['loss']}, Epoch Accuracy: {train_metrics['accuracy'] * 100}")
    
    models = ensemble['models']
    ensemble_params = ensemble['outputs']['params']
    
    train_metrics = ut.generate_results_ensemble(train.X,train.Y,models,ensemble_params,name="Train")
    val_metrics = ut.generate_results_ensemble(validation.X,validation.Y,models,ensemble_params,name="Validation")
    


    ensemble['outputs']['results']['Train']['losses'].append(train_metrics['loss'])
    ensemble['outputs']['results']['Train']['accuracy'].append(train_metrics['accuracy'])
    
    ensemble['outputs']['results']['Validation']['losses'].append(val_metrics['loss'])
    ensemble['outputs']['results']['Validation']['accuracy'].append(val_metrics['accuracy'])
    

    # if val_metrics['accuracy']<last_val_acc:
    #     break
    # else:
    #     last_val_acc = val_metrics['accuracy']
    if epoch%5==0 or epoch == n_epochs-1:
        print(f"saving: {output_name}")
        with open(output_path + output_name + '.pkl', 'wb') as file:
            pickle.dump(ensemble['outputs'],file)



if config['visualisation']['video']:
    ut.plotEpoch(train.X,train.Y,
          ensemble['models'][0],
          plot_states,
          K = train.K,
          key = ensemble['init_rngs'][0],
          name = output_name,
          project_dir = project_dir,
          plot_type='video')

