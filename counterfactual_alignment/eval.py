import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import ast

import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training import train_state
from flax.training import checkpoints
from custom_datasets import AdultDataset, XORDataset
import torch.utils.data as data

import optax

from utilities import numpy_collate, draw_classifier, sigmoid, create_train_state, compute_metrics

from csv import DictWriter

from custom_datasets import XORDataset, GaussianCloudDirection
from custom_models   import SimpleClassifier

setupFile = pd.read_csv('params.csv', header=0,index_col='Parameter')

# XOR, GaussCloud
dataset_select  = str(setupFile.loc['dataset','Value'])
lf              = str(setupFile.loc['loss_function','Value'])
loss_mix        = float(setupFile.loc['loss_mix','Value'])

seed            = int(setupFile.loc['seed','Value'])
batch_size      = int(setupFile.loc['batch_size','Value'])
train_epochs    = int(setupFile.loc['epochs','Value'])
training_points = int(setupFile.loc['training_points','Value'])
n_vec           = int(setupFile.loc['n_vec','Value'])

learning_rate   = float(setupFile.loc['learning_rate','Value'])
momentum        = float(setupFile.loc['momentum','Value'])

f_prefix = ('ckpt_tp'+str(training_points)+
            '_dataclass' + str(setupFile.loc['dataset','Value']) + 
            '_lossfn' + str(setupFile.loc['loss_function','Value']) + 
            '_loss_mix' + str(setupFile.loc['loss_mix','Value']) +
            '_dscheme' + str(setupFile.loc['direction_scheme','Value']) +
            '_nvec'    + str(setupFile.loc['n_vec','Value']) + 
            '_lr' + str(setupFile.loc['learning_rate','Value']) + 
            '_mom' + str(setupFile.loc['momentum','Value']) +
            '_epoch'+ str(train_epochs))

# f_prefix = 'my_model_tp'+str(sys.argv[1])+'_' + str(sys.argv[2])+'_' + str(sys.argv[3]) + '_' 

# Giving the model 8 hidden neurons in between the two input variable (x_1, x_2) and the output label prediction (y_hat)
model = SimpleClassifier(num_hidden=8, num_outputs=1)
# Printing the model shows its attributes
# print(model)

##### Initialise Parameters
#1) Create random input of dataset
seed = 123
learning_rate = 0.05
momentum = 0.8
train_epochs = 100
batch_size = 128

rng = jax.random.PRNGKey(seed)
rng, inp_rng, init_rng = jax.random.split(rng, 3)
model_size = (8,1)
model_io = (8,2)
inp = jax.random.normal(inp_rng, model_io)  # Batch size 8, input size 2

#2) Apply the init function
# Initialize the model
params = model.init(init_rng, inp)

# Choose an optimiser
optimizer = optax.sgd(learning_rate=0.1)

# Create a TrainState which bundles the parameters, the optimizer, and the forward step of the model:
model_state = train_state.TrainState.create(apply_fn=model.apply,
                                            params=params,
                                            tx=optimizer)


trained_state = create_train_state(jax.random.PRNGKey(seed), init_rng, learning_rate, momentum, model_size, model_io)


loaded_model_state = checkpoints.restore_checkpoint(
                                             ckpt_dir='my_checkpoints/',   # Folder with the checkpoints
                                             target=trained_state,
                                             prefix=f_prefix  # Checkpoint file name prefix
                                            )



if dataset_select=='XOR':
    dataset_class = XORDataset
elif dataset_select == 'GaussCloud':
    dataset_class = GaussianCloudDirection
elif dataset_select == 'Adult':
    dataset_class = AdultDataset
else:
    print >> sys.stderr, "Non-existent dataset. Choose XOR, GaussCloud or Adult"
    sys.exit(1)

print("Generating Test Set")
test_dataset = dataset_class(size=500, seed=123, train=False, visualise=False)

print("...and test dataloader")
# drop_last -> Don't drop the last batch although it is smaller than 128
test_data_loader = data.DataLoader(test_dataset,
                                   batch_size=128,
                                   shuffle=False,
                                   drop_last=False,
                                   collate_fn=numpy_collate)


def predict(state, params, data):
    logits = state.apply_fn(params, data).squeeze(axis=-1)
    probabilities = sigmoid(logits)
    pred_labels = (logits > 0).astype(jnp.float32)
    
    return logits, probabilities, pred_labels

def calculate_loss_acc(state, params, batch):
    data_input, labels, direction, direction_label = batch
    # Obtain the logits and predictions of the model for the input data
    
    # logits = state.apply_fn(params, data_input).squeeze(axis=-1)
    # pred_labels = (logits > 0).astype(jnp.float32)
    logits, probs, pred_labels = predict(state, params, data_input)
    
    # Calculate the loss and accuracy
    loss = optax.sigmoid_binary_cross_entropy(logits, labels).mean()
    acc = (pred_labels == labels).mean()
    return loss, acc


def eval_model(state, data_loader):
    all_accs, batch_sizes = [], []
    x_data = np.empty((0,2))
    y_data = np.empty((0), int)

    print("     Calculating evaluation metrics...")
    for batch in data_loader:
        
        data_input, labels, direction, direction_label = batch
        x_data = np.concatenate((x_data,data_input),axis=0)
        
        y_data = np.concatenate((y_data, labels), axis=0)
        batch_acc = eval_step(state, batch)
        all_accs.append(batch_acc)
        batch_sizes.append(batch[0].shape[0])
    # Weighted average since some batches might be smaller

    print("    Generating Figures...")
    
    draw_classifier(predict, state, state.params, x_data, y_data, lims=None)

    acc = sum([a*b for a,b in zip(all_accs, batch_sizes)]) / sum(batch_sizes)
    print(f"Accuracy of the model: {100.0*acc:4.2f}%")
    return acc

print("Evaluation")

@jax.jit
def eval_step(state, data, label):
    logits = SimpleClassifier(*model_size).apply({'params': state.params}, data).squeeze(axis=-1)
    return compute_metrics(logits=logits, labels = label)

def evaluate_model(state, data, label):
    """Evaluate on the validation set."""
    print("     Calculating evaluation metrics...")
    metrics = eval_step(state, data, label)
    metrics = jax.device_get(metrics)  # pull from the accelerator onto host (CPU)
    metrics = jax.tree_map(lambda x: x.item(), metrics)  # np.ndarray -> scalar
    return metrics

test_data = test_dataset.data
test_labels = test_dataset.label

if ast.literal_eval(setupFile.loc['draw_eval','Value']):
    draw_classifier(SimpleClassifier(*model_size).apply, loaded_model_state, test_data, test_labels)


test_metrics = evaluate_model(loaded_model_state, test_data, test_labels)
print(f"Loss: {test_metrics['loss']}, accuracy: {test_metrics['accuracy'] * 100}")

# list of column names 
field_names = ['Model','Accuracy']
  
# Dictionary
dict={'Model':[f_prefix],'Accuracy':[test_metrics['accuracy']]}

# Open your CSV file in append mode
# Create a file object for this file

# columns=['Name', 'ID', 'Department'])
try:
    df = pd.read_csv('test.csv')

    if f_prefix in df['Model'].values:
        # df.loc[df["gender"] == "male", "gender"] = 1
        df.loc[df["Model"] == f_prefix, "Accuracy"] = test_metrics['accuracy']
        # print('INDEX: ',df.iloc[df['Model']==f_prefix])
        # df['Accuracy'].values[df.loc[df['Model']==f_prefix]]=acc
    else:
        df_update = pd.DataFrame(dict,columns = field_names)
        df = pd.concat([df,df_update])

except:
    df = pd.DataFrame(dict,columns = field_names)

df.to_csv('test.csv',index=False)
