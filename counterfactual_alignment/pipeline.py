

import yaml
import pickle
from tqdm import tqdm
import os
os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" # ensure access to homebrew
import numpy as np
import pandas as pd

import seaborn as sns
palette = sns.color_palette("Paired", 6)

import jax
import optax
import torch
import torch.utils.data as data

import counterfactual_alignment as cfa
from counterfactual_alignment import custom_datasets, custom_models, loss_functions, knowledge_functions
from counterfactual_alignment import utilities as ut
import time


class Pipeline:
    def __init__(self, datasets, data_description='', method_description='', overwrite=True, K=None, config=None):
        self.datasets = datasets
        self.data_description = data_description
        self.method_description = method_description
        self.overwrite = overwrite
        self.K = K

        if config is not None:
            self.config = config
            
        else:
            with open("config.yaml", 'r') as file:
                self.config = yaml.unsafe_load(file)

        self.loss_name = self.config['hyperparams']['loss_function']
        self.learning_rate = self.config['hyperparams']['learning_rate']
        self.batch_size = self.config['hyperparams']['batch_size']
        self.seed = self.config['hyperparams']['seed']

        self.output_name = None  # Defer output_name creation until setup

    def setup(self):
        self.key = jax.random.PRNGKey(self.seed)
        self.noise_key, self.sample_key, self.model_key = jax.random.split(self.key, 3)

        self.model = custom_models.SimpleClassifier(*self.config['hyperparams']['model_io'])
        self.optimiser, self.optim_name = self._setup_optimiser()

        self.train_dataloader = data.DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            generator=torch.Generator().manual_seed(self.seed)
        )

        self.trained_state, self.model = ut.create_train_state(
            self.model,
            self.optimiser,
            vector_length=2,
            key=self.model_key
        )

        self.output_name = self._build_output_name()

    def _setup_optimiser(self):
        warmup_steps = 3
        schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(0.0, self.learning_rate, warmup_steps),
                optax.exponential_decay(self.learning_rate, 100, 0.9)
            ],
            boundaries=[warmup_steps]
        )
        adam = optax.adam(self.learning_rate)
        optimiser = adam
        optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0]
        return optimiser, optim_name

    def _build_output_name(self):
        model_name = self.config['hyperparams']['model']
        return "__".join([
            f"MODEL_{model_name}",
            f"OPTIM_{self.optim_name}",
            f"LR_{self.learning_rate}",
            f"BATCHSIZE_{self.batch_size}",
            f"DATA_{self.data_description}",
            f"SIZE_{self.config['data_params']['train_size']}",
            f"LOSS_{self.loss_name}",
            f"ALPHA_{self.config['hyperparams']['loss_mix']}",
            f"METHOD_{self.method_description}"
        ])

    def run(self, n_epochs=10):
        if self.output_name is None:
            self.setup()

        print("Loading and saving to:", self.output_name)
        results = {
            'Train': {'losses': [], 'accuracy': []},
            'Validation': {'losses': [], 'accuracy': []}
        }

        if not self.overwrite:
            try:
                with open(self.output_name + '.pkl', 'rb') as file:
                    res = pickle.load(file)
                self.trained_state = self.trained_state.replace(params=res['params'])
                results = res['results']
                print(f'Model loaded from {self.output_name}')
            except:
                pass

        plot_states = []

        for epoch in tqdm(range(n_epochs)):
            self.trained_state, _ = ut.train_one_epoch(
                self.trained_state,
                self.datasets['train'],
                self.model,
                loss_functions.loss_functions[self.loss_name],
                self.model_key,
                self.config
            )

            plot_states.append(self.trained_state)

            train_metrics = ut.generate_results(
                self.datasets['train']['original'],
                self.model,
                self.trained_state.params,
                name="Train"
            )
            val_metrics = ut.generate_results(
                self.datasets['test'],
                self.model,
                self.trained_state.params,
                name="Validation"
            )

            results['Train']['losses'].append(train_metrics['loss'])
            results['Train']['accuracy'].append(train_metrics['accuracy'])
            results['Validation']['losses'].append(val_metrics['loss'])
            results['Validation']['accuracy'].append(val_metrics['accuracy'])

        self._save_results(results, plot_states, n_epochs)

    def _save_results(self, results, plot_states, n_epochs):
        output = {'results': results, 'params': self.trained_state.params}
        os.makedirs('outputs', exist_ok=True)
        file_path = os.path.join('outputs', self.output_name) + '.pkl'
        with open(file_path, 'wb') as file:
            pickle.dump(output, file)

        print(f"Model saved to /outputs as: {self.output_name}. Trained for {n_epochs} epochs.")

        if self.config['visualisation']['video']:
            ut.plotEpoch(
                self.datasets['test']['X'],
                self.datasets['test']['Y'],
                self.model,
                plot_states,
                plot_type='video',
                name=self.output_name,
                key=self.model_key
            )


# import yaml
# import pickle
# from tqdm import tqdm
# import os
# os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin" # ensure access to homebrew
# import numpy as np
# import pandas as pd

# import seaborn as sns
# palette = sns.color_palette("Paired", 6)

# import jax
# import optax
# import torch
# import torch.utils.data as data

# import counterfactual_alignment as cfa
# from counterfactual_alignment import custom_datasets, custom_models, loss_functions, knowledge_functions
# from counterfactual_alignment import utilities as ut
# import time



# class Pipeline:
#     def __init__(self, datasets, data_description='', method_description='', overwrite=True, K=None):
#         self.datasets = datasets
#         self.data_description = data_description
#         self.method_description = method_description
#         self.overwrite = overwrite
#         self.K = K

#         # Load config
#         self.config_file = "config.yaml"
#         with open(self.config_file, 'r') as file:
#             self.config = yaml.unsafe_load(file)

#         self.loss_name = self.config['hyperparams']['loss_function']
#         self.learning_rate = self.config['hyperparams']['learning_rate']
#         self.batch_size = self.config['hyperparams']['batch_size']
#         self.seed = self.config['hyperparams']['seed']

#         self.key = jax.random.PRNGKey(self.seed)
#         self.noise_key, self.sample_key, self.model_key = jax.random.split(self.key, 3)

#         self.model = custom_models.SimpleClassifier(*self.config['hyperparams']['model_io'])
#         self.optimiser, self.optim_name = self._setup_optimiser()

#         self.train_dataloader = data.DataLoader(
#             datasets['train'],
#             batch_size=self.batch_size,
#             shuffle=True,
#             drop_last=False,
#             generator=torch.Generator().manual_seed(self.seed)
#         )

#         self.trained_state, self.model = ut.create_train_state(
#             self.model,
#             self.optimiser,
#             vector_length=2,
#             key=self.model_key
#         )

#         self.output_name = self._build_output_name()

#     def _setup_optimiser(self):
#         warmup_steps = 3
#         schedule = optax.join_schedules(
#             schedules=[
#                 optax.linear_schedule(0.0, self.learning_rate, warmup_steps),
#                 optax.exponential_decay(self.learning_rate, 100, 0.9)
#             ],
#             boundaries=[warmup_steps]
#         )
#         adam = optax.adam(self.learning_rate)  # Replace with chained if needed
#         optimiser = adam
#         # optim_name = [n for n, v in locals().items() if v is optimiser and n != 'optimiser'][0]
#         optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0] #reads optimiser name from local variables (for file name)
#         return optimiser, optim_name

#     def _build_output_name(self):
#         model_name = self.config['hyperparams']['model']
#         return "__".join([
#             f"MODEL_{model_name}",
#             f"OPTIM_{self.optim_name}",
#             f"LR_{self.learning_rate}",
#             f"BATCHSIZE_{self.batch_size}",
#             f"DATA_{self.data_description}",
#             f"SIZE_{len(self.datasets['train'].tensors[0])}",
#             f"LOSS_{self.loss_name}",
#             f"ALPHA_{self.config['hyperparams']['loss_mix']}",
#             f"METHOD_{self.method_description}"
#         ])

#     def run(self, n_epochs=10):


#         print("Loading and saving to:", self.output_name)
#         results = {
#             'Train': {'losses': [], 'accuracy': []},
#             'Validation': {'losses': [], 'accuracy': []}
#         }

#         if not self.overwrite:
#             try:
#                 with open(self.output_name + '.pkl', 'rb') as file:
#                     res = pickle.load(file)
#                 self.trained_state = self.trained_state.replace(params=res['params'])
#                 results = res['results']
#                 print(f'Model loaded from {self.output_name}')
#             except:
#                 pass

#         plot_states = []

#         for epoch in tqdm(range(n_epochs)):
#             self.trained_state, _ = ut.train_one_epoch(
#                 self.trained_state,
#                 self.train_dataloader,
#                 self.model,
#                 loss_functions.loss_functions[self.loss_name],
#                 self.model_key
#             )

#             plot_states.append(self.trained_state)

#             train_metrics = ut.generate_results(
#                 self.datasets['train'].tensors,
#                 self.model,
#                 self.trained_state.params,
#                 name="Train"
#             )
#             val_metrics = ut.generate_results(
#                 self.datasets['test'].tensors,
#                 self.model,
#                 self.trained_state.params,
#                 name="Validation"
#             )

#             results['Train']['losses'].append(train_metrics['loss'])
#             results['Train']['accuracy'].append(train_metrics['accuracy'])
#             results['Validation']['losses'].append(val_metrics['loss'])
#             results['Validation']['accuracy'].append(val_metrics['accuracy'])

#         self._save_results(results, plot_states, n_epochs)

#     def _save_results(self, results, plot_states, n_epochs):
#         output = {'results': results, 'params': self.trained_state.params}
#         os.makedirs('outputs', exist_ok=True)
#         file_path = os.path.join('outputs', self.output_name) + '.pkl'
#         with open(file_path, 'wb') as file:
#             pickle.dump(output, file)

#         print(f"Model saved to /outputs as: {self.output_name}. Trained for {n_epochs} epochs.")

#         if self.config['visualisation']['video']:
#             ut.plotEpoch(
#                 self.datasets['test'].tensors[0],
#                 self.datasets['test'].tensors[1],
#                 self.model,
#                 plot_states,
#                 plot_type='video',
#                 name=self.output_name,
#                 key=self.model_key
#             )


# # def pipeline(datasets,data_description='',method_description='',overwrite=True, K = None):
# #     """
# #     Load tunable parameters from config file 
# #     """
# #     config_file = "config.yaml" #assuming local yaml file
# #     with open(config_file,'r') as file:
# #         config = yaml.unsafe_load(file)

# #     overwrite = True

# #     loss_name = config['hyperparams']['loss_function']
# #     print(loss_name)
# #     learning_rate = config['hyperparams']['learning_rate']
# #     batch_size = config['hyperparams']['batch_size']

# #     """
# #     Define learning rate schedule
# #     """

# #     #simple optimiser
# #     adam = optax.adam(learning_rate=learning_rate)

# #     #complex optimiser
# #     warmup_steps = 3

# #     final_lr = 1e-3

# #     schedule = optax.join_schedules(
# #         schedules=[
# #             optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=warmup_steps),
# #             optax.exponential_decay(init_value=learning_rate, transition_steps=100, decay_rate=0.9)
# #         ],
# #         boundaries=[warmup_steps]
# #     )

# #     chained_optax = optax.chain(
# #         optax.clip_by_global_norm(1.0),  # Clip gradients
# #         optax.adam(schedule)
# #     )
# #     optimiser = adam

# #     optim_name = [oname for oname in [name for name, value in locals().items() if value is optimiser] if oname != 'optimiser'][0] #reads optimiser name from local variables (for file name)

# #     """
# #     Initialise Parameters
# #     """

# #     seed = seed=config['hyperparams']['seed']
# #     key = jax.random.PRNGKey(seed)
# #     noise_key,sample_key,model_key = jax.random.split(key,3)


    
# #     train_dataloader = data.DataLoader(datasets['train'], batch_size=batch_size, shuffle=False, drop_last=False, generator=torch.Generator().manual_seed(seed))

# #     model = custom_models.SimpleClassifier(*config['hyperparams']['model_io'])
# #     trained_state, model = ut.create_train_state(model,optimiser,vector_length=2, key=model_key)

# #     model_name = config['hyperparams']['model']
# #     output_name = "__".join([f"MODEL_{model_name}",
# #                              f"OPTIM_{optim_name}"])

# #     output_name = "__".join([
# #             f"MODEL_{model_name}",
# #             f"OPTIM_{optim_name}",
# #             f"LR_{learning_rate}",
# #             f"BATCHSIZE_{batch_size}",
# #             f"DATA_{data_description}",
# #             f"SIZE_{len(datasets['train'].tensors[0])}",
# #             f"LOSS_{loss_name}",
# #             f"ALPHA_{config['hyperparams']['loss_mix']}",
# #             f"SIZE_{config['data_params']['train_size']}",
# #             f"METHOD_{method_description}"
# #         ])

# #     print("Loading and saving to : ", output_name)


# #     results= {
# #                     'Train':{'losses':[],'accuracy':[]},
# #                     'Validation':{'losses':[],'accuracy':[]}}

# #     if not overwrite:
# #         try:
# #             print('Test')
# #             with open(output_name+'.pkl', 'rb') as file: ## remove this line to load model
# #                 res = pickle.load(file)
# #             print('Test2')
# #             trained_state.replace(params = res['params'])
            
# #             results = res['results']

# #             print(f'Model loaded from {output_name}')
            
# #         except:
# #             pass
            

# #     plot_states = []
# #     n_epochs = 10

# #     for epoch in tqdm(range(n_epochs)):
# #         trained_state, train_metrics = ut.train_one_epoch(trained_state, train_dataloader,  
# #                                                         # model, loss_functions['direction_interactive_vectorized'])
# #                                                     #  model, loss_functions['direction_interactive'])
# #                                                     # model, loss_functions['direction_interactive2'])
# #                                                     #  model, loss_functions['gradient_supervision'],rng)
# #                                                     #  model, loss_functions['direction'],rng)
# #                                                     model, loss_functions.loss_functions[loss_name],model_key)
        
# #         plot_states.append(trained_state)

# #         train_metrics = ut.generate_results(datasets['train'].tensors,model,trained_state.params,name="Train")
# #         val_metrics = ut.generate_results(datasets['test'].tensors,model,trained_state.params,name="Validation")
        
# #         results['Train']['losses'].append(train_metrics['loss'])
# #         results['Train']['accuracy'].append(train_metrics['accuracy'])
        
# #         results['Validation']['losses'].append(val_metrics['loss'])
# #         results['Validation']['accuracy'].append(val_metrics['accuracy'])
        
# #     output = {'results':results,
# #           'params':trained_state.params}

# #     total_epochs = len(results['Train']['accuracy'])

# #     print(f'Model saved to /outputs as:\n\n{output_name}. \nTrained for {n_epochs} epochs (Total: {total_epochs})')
# #     os.makedirs('outputs',exist_ok=True)
# #     with open(os.path.join('outputs',output_name) + '.pkl', 'wb') as file:
# #         pickle.dump(output,file)

# #     if config['visualisation']['video']:
        
# #         ut.plotEpoch(
# #             datasets['test'].tensors[0],
# #             datasets['test'].tensors[1],
# #             model,
# #             plot_states,
# #             plot_type='video',
# #             name = output_name,
# #             key = model_key)
        
    