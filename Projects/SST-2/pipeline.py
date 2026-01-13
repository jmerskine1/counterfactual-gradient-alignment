# Pipeline:
#  Define a neural net
#  Create a dataset to train on. Standard classifier example, or maybe XOR (or both)
#  Create a function script which enables either Cross-entropy or Direction loss function [ESSENTIAL]
#  Training model on the dataset on set of training points (very few -> 100% accuracy) [ESSENTIAL]
## Standard libraries
import numpy as np
import random
import pickle
import yaml
import matplotlib.pyplot as plt
## Progress bar
from tqdm.auto import tqdm

#ML Libraries
import jax
import optax
import torch

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
import counterfactual_alignment.utilities as utils
from counterfactual_alignment.utilities import (visualise_classes, mnist_collate_beta, sst_collate, select_informative_samples,
                        reduce_dataset, compute_metrics, generate_results, generate_results_ensemble, create_train_state, boundary_filter, save_stats, train_one_epoch, combine_datasets)


# Choose a categorical colormap
CMAP = plt.get_cmap("tab10")


# Get the directory where THIS script is located
project_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-oc", default='config.yaml', help="Config file name")
parser.add_argument("--note", "-N", default="", help="add note to output name")
parser.add_argument("--loss", "-L", default=None, help="Loss function to use (overrides config file)")
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

print(f"Loading dataset: \n{data_name} from \n{data_folder}\n")

with open(os.path.join(data_folder,data_name)+'.pkl', 'rb') as file:
    datasets = pickle.load(file)

n_classes = datasets['n_classes']

output_path = project_dir + "/model_outputs/" + data_name + "/"
os.makedirs(output_path, exist_ok=True)

"""
Initialise Parameters
"""


n_models = config['hyperparams']['n_models']
n_epochs = config['hyperparams']['epochs']
overwrite = True

n_vectors = len(datasets['train']['original']['X'][0])
train_size = len(datasets['train']['original']['Y'])

rng = jax.random.PRNGKey(42)
rng, inp_rng, init_rng, dropout_rng, embedding_rng = jax.random.split(rng, 5)

"""
Model Parameters
"""
if args.loss is not None:
    config['hyperparams']['loss_function'] = args.loss

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

adabelief = optax.adabelief(
    learning_rate=learning_rate,
    b1=0.9,
    b2=0.999,
    eps=1e-16,
)

scheduled_adadelta = optax.adadelta(learning_rate=learning_rate_schedule, weight_decay=0.05)

optimiser = adabelief
# optimiser = adadelta
# optimiser = adam_opt
# optimiser = sgd_opt
# optimiser = scheduled_adadelta
# optimiser = adamw

optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]

# model = SimpleClassifier(8,1)
# model = MLP(64,1)
embedding_size = config['hyperparams'].get('embedding_size',64)
ensemble = {
            # 'models':[BagOfWordsClassifier(20000,50)]*n_models,
            # 'models': [SentimentModel(20000,50)]*n_models,
            'models':[
                custom_models[config['hyperparams']['model']](
                    num_classes=n_classes,
                    embedding_size=embedding_size)
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

if config['data_params']['control']:
    full_training_set = customDataset(datasets['control']['original'])
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

if active_sampling:
    print(f"Active sampling enabled. Initialising with subset of data. Original train size: {full_training_set.X.shape[0]}")
    init_samplesize = config['data_params']['init_samplesize']
    # Set deterministic random seed
    np_rng = np.random.default_rng(seed=42)
    # Sample indices without replacement
    subsample_indices = np_rng.choice(np.arange(train_size), size=init_samplesize, replace=False)
    unsampled = set(np.arange(0,train_size)) - set(subsample_indices)
    training_set = full_training_set.subset(subsample_indices)
else:
    training_set = full_training_set

print(f"Training on {len(training_set)} samples.")
seed = 42

def seed_worker(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

generator = torch.Generator()
generator.manual_seed(seed)

train_original_data_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=sst_collate,
    drop_last=True,
    worker_init_fn=seed_worker,
    generator=generator
)



validation = customDataset(datasets['dev']['original'])
validation_data_loader = data.DataLoader(validation, batch_size=batch_size, shuffle=True, collate_fn=sst_collate, drop_last=True)

test_original = customDataset(datasets['test']['original'])
test_original_data_loader = data.DataLoader(test_original, batch_size=batch_size, shuffle=True, collate_fn=sst_collate, drop_last=True)


output_name = (f"MODEL_ENSEMBLE_{n_models}_{model_name}__"
               f"emb{config['hyperparams']['embedding_size']}_"
               f"OPTIM_{optim_name}__LR_{learning_rate}__BATCHSIZE_{batch_size}__"
               f"trainsize_{train_size}__active_{active_sampling}_smplsize_{sample_size}_div{config['data_params']['diversity']}_"
               f"LOSS_{config['hyperparams']['loss_function']}_mix_{config['hyperparams']['loss_mix']}"+args.note)

print("Loading and saving to : ", output_name)

for i in range(n_models):
    trained_state = create_train_state(ensemble['models'][i],optimiser,vector_length=n_vectors, key=ensemble['init_rngs'][i])
    # trained_state, model = create_train_state(ensemble['models'][i],ensemble['init_rngs'][i],optimiser,batch_size=batch_size,vector_length=n_vectors)
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

if config['visualisation']['visualise_embeddings']:
    from matplotlib.widgets import Slider
    plt.ion()  # interactive mode on

    # ----------------------------------------
    # CONFIG
    # ----------------------------------------
    DATASET_NAMES = ["train", "test"]

    # ----------------------------------------
    # Storage: dataset_name -> list of 2D arrays (one per epoch)
    # e.g. history["train"][epoch] = (N,2) reduced embeddings
    # ----------------------------------------
    history = {name: [] for name in DATASET_NAMES}
    plots = {}   # name -> (fig, ax, scatter, slider)

    
    # ----------------------------------------
    # Create a dataset window (called once per dataset)
    # ----------------------------------------
    def create_dataset_window(name):
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(f"{name} embeddings")

        unique_labels = np.unique(validation.Y)
        # initial empty scatter
        scatter = ax.scatter([], [])
        ax.set_title(f"{name} – epoch 0")
        # Use BoundaryNorm to map label indices to colors
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_classes+0.5), ncolors=n_classes)
        sm = plt.cm.ScalarMappable(cmap=CMAP, norm=norm)
        sm.set_array(unique_labels)

        cbar = plt.colorbar(sm, ax=ax, ticks=np.arange(n_classes))
        cbar.set_label("Labels")
        # Optionally set tick labels to actual labels
        cbar.set_ticklabels([str(l) for l in unique_labels])

        # slider axis
        slider_ax = fig.add_axes([0.15, 0.05, 0.7, 0.04])
        slider = Slider(slider_ax, "Epoch", 0, 0, valinit=0, valstep=1)

        # store
        plots[name] = (fig, ax, scatter, slider)

        # callback
        def on_slider_change(val):
            epoch = int(val)
            update_plot_to_epoch(name, epoch)

        slider.on_changed(on_slider_change)

    import matplotlib.colors as mcolors
    # ----------------------------------------
    # Update the plot for a dataset to a given epoch
    # ----------------------------------------
    def update_plot_to_epoch(name, epoch):
        fig, ax, scatter, slider = plots[name]

        pts,labels = history[name][epoch]

        # convert labels → colors
        unique_labels = np.unique(labels)
        color_map = {l: CMAP(i % 10) for i, l in enumerate(unique_labels)}
        colors = np.array([color_map[l] for l in labels])
        scatter.set_offsets(pts)
        scatter.set_color(colors)
        
        

        ax.set_title(f"{name} – epoch {epoch}")
        # Manually set axis limits from pts (relim won't handle PathCollection)
        utils.set_axes_limits_from_points(ax, pts, margin_ratio=0.06)

        ax.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()

    # ----------------------------------------
    # Called every epoch: compute embedding + store + update latest view
    # ----------------------------------------
    def update_dataset_epoch(name, embeddings,labels):
        pts = utils.reduce_dim(embeddings,method="tsne")
        history[name].append((pts,labels))

        fig, ax, scatter, slider = plots[name]

        # update slider max range
        max_epoch = len(history[name]) - 1
        slider.valmax = max_epoch
        slider.ax.set_xlim(0, max_epoch)

        # jump the slider to the latest epoch
        slider.set_val(max_epoch)

        # update plot
        update_plot_to_epoch(name, max_epoch)

    for name in DATASET_NAMES:
            create_dataset_window(name)

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
    


    ensemble['outputs']['results']['Train']['losses'].append(train_metrics['loss'])
    ensemble['outputs']['results']['Train']['accuracy'].append(train_metrics['accuracy'])
    
    ensemble['outputs']['results']['Validation']['losses'].append(val_metrics['loss'])
    ensemble['outputs']['results']['Validation']['accuracy'].append(val_metrics['accuracy'])
    
    ensemble['outputs']['results']['Test Original']['losses'].append(test_o_metrics['loss'])
    ensemble['outputs']['results']['Test Original']['accuracy'].append(test_o_metrics['accuracy'])
    
    if config['visualisation']['visualise_embeddings']:
        for name in DATASET_NAMES:
            # generate fake embeddings for all datasets
        
            
            X = None
            Y = None
            if name == 'train':
                X = training_set.X
                Y = training_set.Y
            elif name == 'val':
                X = validation.X
                Y = validation.Y
            elif name == 'test':
                X = test_original.X
                Y = test_original.Y
            
            models = ensemble['models']
            params = ensemble['outputs']['params']
            
            # Get embeddings from ensemble
            num_models = len(models)
            num_samples = X.shape[0]
            # Run one model to inspect shape
            sample_logits, _ = models[0].apply({'params': params[0]}, np.array(X), train=False)
        
            # Multi-class
            num_classes = sample_logits.shape[-1]
            logits = np.zeros((num_models, num_samples, num_classes))
            ensemble_embeddings = []
            for i, (model, param) in enumerate(zip(models, params)):
                logits[i, :, :], embeddings = model.apply({'params': param}, np.array(X), train=False)
                ensemble_embeddings.append(embeddings)

            ensemble_embeddings = np.mean(np.stack(ensemble_embeddings), axis=0)

        
            update_dataset_epoch(name, ensemble_embeddings,Y)

    

    # if val_metrics['accuracy']<last_val_acc:
    #     break
    # else:
    #     last_val_acc = val_metrics['accuracy']
    if epoch%5==0 or epoch == n_epochs-1:
        print(f"saving: {output_name}")
        with open(output_path + output_name + '.pkl', 'wb') as file:
            pickle.dump(ensemble['outputs'],file)

    if active_sampling and epoch%2==0 and epoch > 1 and len(training_set) < train_size:
        print("Active learning: selecting new samples...")
        print(len(unsampled))

        ensemble['train_states'] = [
            utils.reset_optimizer(ts, optimiser, m, k, n_vectors)
            for ts, m, k in zip(ensemble['train_states'], ensemble['models'], ensemble['init_rngs'])
        ]
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
        probs = np.array(jax.nn.softmax(logits))

        new_indices = select_informative_samples(probs, embeds, k=sample_size, diversity_weight=config['data_params']['diversity'])
        unsampled = unsampled - set(new_indices)
        
        new_subset = remainder.subset(new_indices)
        # new_subset = [remainder[i] for i in new_indices]
        # print(new_subset['X'])
        training_set = training_set.combine(new_subset)
        print(f"New training set size: {len(training_set)}")
        # new_subset = combine_datasets(initial_subset,new_subset)
        train_original_data_loader = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, collate_fn=sst_collate, drop_last=True)
        # Move selected samples from unlabeled to training

while config['visualisation']['visualise_embeddings']:
        for fig, ax, scatter, slider in plots.values():
            fig.canvas.flush_events()
        plt.pause(0.01)