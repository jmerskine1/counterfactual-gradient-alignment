from torch import normal
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy.stats import multivariate_normal as mvn

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from flax import traverse_util
import flax
from flax import linen as nn

from tqdm.auto import tqdm

import random
import seaborn as sns
import sys
## Imports for plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib import colormaps

import time
import math
import subprocess
import glob
import os
import json
import csv
from pathlib import Path
from functools import partial

import counterfactual_alignment.custom_models as cm
from counterfactual_alignment.custom_models import custom_models

from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

# FAT Forensics Counterfactual Explainer
import fatf.transparency.predictions.counterfactuals as fatf_cf

from scipy.sparse import hstack, vstack

import tensorflow as tf

def pad_sequences_tf(sequences, maxlen, padding='post', value=0):
    """ Pad sequences using TensorFlow/Keras utility.
    
    Args:
        sequences (list of list of int): List of sequences.
        maxlen (int): Desired maximum length of the padded sequences.
        padding (str): 'post' to pad after the sequences, 'pre' to pad before.
        value (int): Padding value.
    
    Returns:
        np.array: An array of padded sequences.
    """
    return tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=padding,
        value=value
    )


def combine_datasets_archive(data1,data2):
    
    X = {}
    for key in data1['X']:
        X[key] = generic_append(data1['X'][key],data2['X'][key])
    
    K = {}
    for key in data1['K']:
        K[key] = generic_append(data1['K'][key],data2['K'][key])

    data3 = {'X':X,
             'Y':np.append(data1['Y'],data2['Y']),
             'K':K}

    return data3

def combine_datasets(data1,data2):
    
    K = {}
    for key in data1['K']:
        K[key] = generic_append(data1['K'][key],data2['K'][key])

    data3 = {'text':np.append(data1['text'],data2['text']),
             'X':np.append(data1['X'],data2['X'],axis=0),
             'Y':np.append(data1['Y'],data2['Y']),
             'K':K}
    
    return data3


def generic_append(item1, item2):
    if isinstance(item1, list):
        item3 = item1.copy()
        item3.extend(item2)
        
    elif isinstance(item1, np.ndarray):
        item3 = np.append(item1, item2,axis=0)
        
    else:
        raise TypeError("Unsupported container type. Expected list or numpy.ndarray.")
    
    
    return item3

def reduce_dataset(data,ratio):
    np.random.seed(2)
    size = int(len(data['Y'])*ratio)
    print(f"Dataset reduced from {len(data['Y'])} to {size}.")
    inds = np.random.randint(0,len(data['Y']),size=size)
    
    def reduce_data(sub_data):
        reduced_data = {}
        for key, values in sub_data.items():
            # Directly use advanced indexing
            if isinstance(values, list):
                reduced_data[key] = [values[i] for i in inds]
            elif isinstance(values, np.ndarray):
                reduced_data[key] = values[inds]
            else:
                raise ValueError("Unsupported data type.")
        return reduced_data
    
    X = reduce_data(data['X'])
    K = reduce_data(data['K'])
    
    return {'X':X,
             'Y':[data['Y'][i] for i in inds],
             'K':K}

def visualise_samples(data, label):
    data_0 = data[label == 0]
    data_1 = data[label == 1]

    plt.figure(figsize=(4,4))
    plt.scatter(data_0[:,0], data_0[:,1], edgecolor="#333", label="Class 0")
    plt.scatter(data_1[:,0], data_1[:,1], edgecolor="#333", label="Class 1")
    plt.title("Dataset samples")
    plt.ylabel(r"$x_2$")
    plt.xlabel(r"$x_1$")
    plt.legend()

# This collate function is taken from the JAX tutorial with PyTorch Data Loading
# https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html

def numpy_collate(batch):
    if isinstance(batch[0], jnp.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)

def imdb_collate(batch):
    X = [b['X'] for b in batch]
    text = [b['text'] for b in batch]
    labels = [b['Y'] for b in batch]
    k_txt = [b['K']['text'] for b in batch]
    k_vec = [b['K']['vector'] for b in batch]
    k_lab = [b['K']['label'] for b in batch]
    k_mag = [b['K']['magnitude'] for b in batch]
    
    knowledge = {'text':k_txt,'vector':jnp.stack([jnp.stack(k_v) for k_v in k_vec]),'label':k_lab,'magnitude':k_mag}

    return{'text':text,'X':X,'Y':labels,'K':knowledge}

def custom_collate_2D(batch):
    
    x_vec = [b['X'] for b in batch]
    
    labels = [b['Y'] for b in batch]
    
    k_vec = [b['K']['vector'] for b in batch]
    k_lab = [b['K']['label'] for b in batch]
    k_mag = [b['K']['magnitude'] for b in batch]
    
    X = jnp.stack(x_vec)
    labels = jnp.stack(labels)
    # knowledge = {'text':k_txt,'vector':jnp.stack(k_vec),'label':k_lab,'magnitude':k_mag}
    knowledge = {'vector':jnp.stack([jnp.stack(k_v) for k_v in k_vec]),'label':k_lab,'magnitude':k_mag}

    return{'X':X,'Y':labels,'K':knowledge}


def get_rand_vec(dims):
    x = np.random.standard_normal(dims)
    r = np.sqrt((x*x).sum())
    return x / r


def get_unit_vec(p1,p2):
      x1 = p1[0]
      x2 = p2[0]
      y1 = p1[1]
      y2 = p2[1]

      vec_x = x2 - x1
      vec_y = y2 - y1
      
      distance = np.sqrt(abs(vec_x)**2 + abs(vec_y)**2)

      if distance == 0:
          Warning(f"Returning difference instead of euclidean distance as at least one dimension is unchanged: \nPoint 1:{p1}, Point 2:{p2}")
          return np.array((vec_x,vec_y)), distance

      return np.array([vec_x,vec_y])/distance, distance



def expand_data(dataclass):
        n_vec = dataclass.n_vec
        x = []
        y = []
        du = []
        dv = []
        dl = []
        dd = []

        for i,d in enumerate(dataclass.data):
            x.extend([d[0]]*n_vec)
            y.extend([d[1]]*n_vec)
            try:
              du.extend(dataclass.directions[i][:,0])
              dv.extend(dataclass.directions[i][:,1])          #  = np.append(d,[dataclass.directions[i][:,0],dataclass.directions[i][:,1]])
              dl.extend(dataclass.direction_label[i][:])
              dd.extend(dataclass.direction_distance[i][:])
            except:
              du.extend([0])
              dv.extend([0])
              dl.extend([0])
              dd.extend([0])
            
        
        data = jnp.column_stack((x,y)) 
        directions = jnp.column_stack((du,dv))
        direction_labels = jnp.array(dl)
        direction_distance = jnp.array(dd)

        return data, directions, direction_labels, direction_distance

def visualise_classes(dataset,knowledge=True):

  scale = 1
  fig = plt.figure(figsize = (5,5))
  ax = fig.add_subplot(111)
  # fig.set_size_inches(18.5, 10.5)
  X = dataset.X['vector']
  x,y = X[:,0],X[:,1]
  labels = np.unique(dataset.Y)

  n = len(labels)
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])
  
  handles = []

  if knowledge:  
    normal_pal = sns.color_palette("Set1",(n+1)*2)
    pastel_pal = sns.color_palette("Pastel1",(n+1)*2)
    normal_pal.as_hex()
    pastel_pal.as_hex()

    for c_i,c in enumerate([int(l) for l in labels]):
      
      c_i_n = int(abs(c_i - 1))

      normal_patch = mpatches.Patch(color=normal_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = -1$')
      pastel_patch = mpatches.Patch(color=pastel_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = 1$')      
      handles.extend((normal_patch,pastel_patch))
    
    for i in range(np.shape(dataset.K['vector'])[0]):
        for j in range(np.shape(dataset.K['vector'])[1]):
    
            u = dataset.K['vector'][i,j,0]*dataset.K['magnitude'][i,j]
            v = dataset.K['vector'][i,j,1]*dataset.K['magnitude'][i,j]
    
            ax.quiver(x[i],y[i],u,v,angles='xy', scale_units = 'xy',
                                          color=pastel_pal[dataset.Y[i]],width=1/200,alpha=1.0,headlength=4,headwidth=4,scale=1)
  ax.legend(handles=handles)

  ax.scatter(x,y,c=dataset.Y, cmap=cm_bright, edgecolors='k')
  plt.show()

  return fig, ax





from matplotlib.lines import Line2D

def plot_from_results_file(results, xlims=None, ylims=None, loss=False, labels=None):
    if type(results) != list:
        results = [results]

    c_palette = sns.color_palette('hls', len(results))
    linestyles = [':', '-.', '--', '-']

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(top=1.0)

    lambda_colors = {}  # Store λ : color mapping
    custom_legend = [
        Line2D([0], [0], color='black', linestyle='dotted', label='Train'),
        Line2D([0], [0], color='black', linestyle='dashdot', label='Validation'),
    ]

    for i, results_set in enumerate(results):
        col = c_palette[i]
        count = 0

        for name, res in results_set.items():
            label = name
            if labels and i < len(labels):
                label = f'{labels[i]} - {name}'

            accuracy = res['accuracy']
            if len(accuracy) == 0:
                continue

            x = range(len(accuracy))
            linestyle = linestyles[count % len(linestyles)]
            line = ax.plot(x, accuracy, linestyle=linestyle, label=label, color=col)

            # Max accuracy marker
            max_acc = np.max(accuracy)
            max_idx = np.argmax(accuracy)
            ax.plot(max_idx, max_acc, 'o', color=col, markersize=8)
            ax.text(1, max_acc, f'{max_acc*100:.1f}%', fontsize=10, 
                    verticalalignment='bottom', horizontalalignment='right', color=col)

            # Scatter Loss
            if 'train' in name.lower() and loss:
                losses = res['losses']
                x_loss = range(len(losses))
                ax.scatter(x_loss, losses, marker='+', s=20, color=col, alpha=0.5)

            # Track unique λ colors
            if labels and i < len(labels):
                lambda_val = labels[i]
                if lambda_val not in lambda_colors:
                    lambda_colors[lambda_val] = col

            count += 1

    # Extend custom legend with unique λ colors
    for lam, color in lambda_colors.items():
        custom_legend.append(Line2D([0], [0], color=color, linestyle='-', label=f'λ = {lam}'))

    ax.legend(handles=custom_legend, title="Legend", loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')

    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)

    plt.tight_layout()
    plt.show()
    return fig, ax


def visualise_classes_archive(data):

  scale = 0.5
  fig = plt.figure(figsize = (5,5))
  ax = fig.add_subplot(111)
  # fig.set_size_inches(18.5, 10.5)  
  
  normal_pal = sns.color_palette("Set1",(len(data.classes)+1)*2)
  pastel_pal = sns.color_palette("Pastel1",(len(data.classes)+1)*2)
  normal_pal.as_hex()
  pastel_pal.as_hex()

  handles = []
  for c_i,c in enumerate(data.classes):
    
    x   = []
    y   = []
    u   = []
    v   = []
    col = []
    c_i_n = int(abs(c_i - 1))
    cols = [normal_pal[int(c_i_n)],pastel_pal[int(c_i_n)]]
    normal_patch = mpatches.Patch(color=normal_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = -1$')
    pastel_patch = mpatches.Patch(color=pastel_pal[int(c_i_n)], label=f'Class {c_i+1} | $s = 1$')
    
    handles.extend((normal_patch,pastel_patch))
    if len(c.directions)==0:
      n = 0
    else:
      n = len(c.directions[0])

    # go through each label, if -1, change color?
    for i,d in enumerate(c.data):
      x.extend([d[0]]*n)
      y.extend([d[1]]*n)
      u.extend(c.directions[i][:,0])
      v.extend(c.directions[i][:,1])
      col.extend([cols[0] if dl < 0 else cols[int(dl)] for dl in c.direction_label[i][:]])
    
    ax.quiver(x,y,np.array(u)*scale,np.array(v)*scale,color=col,width=1/200, scale=10,alpha=0.5,headlength=4,headwidth=4)
  ax.legend(handles=handles)
  # plt.show()

  return fig, ax


def draw_classifier(predict, state, X_train, y_train, lims=None):
                    # ({'params': loaded_model_state.params}, data)
  fig, ax = plt.subplots()
  h = 0.1
  if lims is None:
    x_min, x_max = X_train[:, 0].min() - .5, X_train[:, 0].max() + .5
    y_min, y_max = X_train[:, 1].min() - .5, X_train[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
  xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], h),
                      np.arange(lims[1][0], lims[1][1], h))
  logits = predict({'params': state.params}, np.stack([xx.ravel(), yy.ravel()]).T)
  probabilities = sigmoid(logits)
  Z = probabilities[:,1]
  Z_r = Z.reshape(xx.shape)
  x_max = []
  y_max = []
  x_p5 = []
  y_p5 = []
  
  
  for i in range(len(xx)):
    max_idx = np.argmax(Z_r[i])
    nearest_idx = (np.abs(Z_r[i] - 0.5)).argmin()
    x_max.append(xx[i][max_idx])
    y_max.append(yy[i][max_idx])
    x_p5.append(xx[i][nearest_idx])
    y_p5.append(yy[i][nearest_idx])
    
  cm = plt.cm.RdBu
  cm_bright = ListedColormap(['#FF0000', '#0000FF'])
  im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)
  fig.colorbar(im, ax=ax)
  
  ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
  ax.scatter(x_max,y_max,facecolor='k',label='Max Probability',marker='1')
  ax.scatter(x_p5,y_p5,facecolor='orange',label = 'Probability=0.5',marker='2')
  ax.legend()
  ax.set_xlim(lims[0])
  ax.set_ylim(lims[1])
  plt.show()

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def compute_metrics(logits, labels):

    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits, labels))
    pred_labels = (nn.sigmoid(logits) > 0.5).astype(np.float32)
    accuracy = (pred_labels == labels).mean()
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics

def generate_results(data,model,params,name='Untitled',verbose=False):
        X = data['X']
        Y = data['Y']
        logits = model.apply({'params':params},np.array(X))
        metrics = compute_metrics(logits,np.array(Y))
        if verbose:
            print(f"{name} Loss: {metrics['loss']}, {name} Accuracy: {metrics['accuracy'] * 100}")
        return metrics

def generate_results_ensemble(X,Y,models,params,name='Untitled'):
        logits = np.zeros((len(models),len(X)))
        
        for i, (model, param) in enumerate(list(zip(models,params))):
            logits[i,:],_ = model.apply({'params':param},np.array(X),train=False)
        
        logits = np.mean(logits,axis=0)
        metrics = compute_metrics(logits,np.array(Y))
        print(f"{name} Loss: {metrics['loss']}, {name} Accuracy: {metrics['accuracy'] * 100}")
        return metrics

class MyTrainState(train_state.TrainState):
    batch_stats: flax.core.FrozenDict

def create_train_state_batch(model,init_rng,opt,batch_size = 128,vector_length=32):
    
    key = jax.random.PRNGKey(42)
    main_key, dropout_key = jax.random.split(key, 2)
    # model = custom_models['GPTattempt'](*(batch_size,1))


    # model = custom_models['simple'](*(8,1))
    init_rngs =  {'params': main_key, 'dropout': dropout_key}
    
    dummy_input = jax.random.randint(key, (batch_size, vector_length), minval=0, maxval=20000)
    
    # inp = jax.random.normal(main_key, (batch_size,vector_length))

    # params = simpleModel.init(key, np.ones([1,*model_size]))['params']
    # params = model.init(init_rngs, jax.random.normal(init_rng, (batch_size,n_input)))['params']
    variables = model.init(init_rngs, dummy_input)
    # params = variables['params']
    
    
    # TrainState is a simple built-in wrapper class that makes things a bit cleaner
    # return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt), model
    # use MyTrainState for batch_norm
    return MyTrainState.create(apply_fn=model.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=opt), model

def balanced_class_sample(X, y, n_per_class, seed=None, return_indices=False):
    from collections import defaultdict
    import random

    if seed is not None:
        random.seed(seed)

    class_to_indices = defaultdict(list)
    for idx, label in enumerate(y):
        class_to_indices[label].append(idx)

    sampled_indices = []
    for label, indices in class_to_indices.items():
        if len(indices) < n_per_class:
            raise ValueError(f"Not enough samples in class {label} to draw {n_per_class}")
        sampled_indices.extend(random.sample(indices, n_per_class))

    sampled_indices = sorted(sampled_indices)
    all_indices = set(range(len(y)))
    remaining_indices = sorted(all_indices - set(sampled_indices))

    if return_indices:
        return sampled_indices, remaining_indices

    X_sampled = np.array([X[i] for i in sampled_indices])
    y_sampled = np.array([y[i] for i in sampled_indices])

    X_remaining = np.array([X[i] for i in remaining_indices])
    y_remaining = np.array([y[i] for i in remaining_indices])

    return X_sampled, y_sampled, X_remaining, y_remaining
# def create_train_state(model,init_rng,opt,batch_size = 128,vector_length=32):
    
#     key = jax.random.PRNGKey(42)
#     main_key, dropout_key = jax.random.split(key, 2)
#     # model = custom_models['GPTattempt'](*(batch_size,1))


#     # model = custom_models['simple'](*(8,1))
#     init_rngs =  {'params': main_key, 'dropout': dropout_key}
    
#     # dummy_input = jax.random.randint(key, (batch_size, vector_length), minval=0, maxval=20000)
#     dummy_input = jax.random.randint(key, (batch_size, vector_length), minval=0, maxval=20000)

#     # inp = jax.random.normal(main_key, (batch_size,vector_length))

#     # params = simpleModel.init(key, np.ones([1,*model_size]))['params']
#     # params = model.init(init_rngs, jax.random.normal(init_rng, (batch_size,n_input)))['params']
#     variables = model.init(init_rngs, dummy_input)
#     # params = variables['params']
    
    
#     # TrainState is a simple built-in wrapper class that makes things a bit cleaner
#     # return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt), model
#     # use MyTrainState for batch_norm
#     return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'],  tx=opt), model

def create_train_state(model, opt, vector_length=768, embedding_dim=50, key = None):
    if key == None:
        key = jax.random.PRNGKey(42)
    
    main_key, dropout_key = jax.random.split(key)

    init_rngs = {'params': main_key, 'dropout': dropout_key}

    # dummy_input = jax.random.normal(main_key, (1, vector_length, embedding_dim))
    # dummy_input = jax.random.normal(main_key, (1, vector_length))
    dummy_input = jax.random.randint(main_key, (1, vector_length),minval=0, maxval=20000)
    
    variables = model.init(init_rngs, dummy_input)

    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=opt
    ), model


# def create_train_state(model,init_rng,opt,batch_size = 8,vector_length=768):
    
#     key = jax.random.PRNGKey(42)
#     main_key, dropout_key = jax.random.split(key, 2)
#     # model = custom_models['GPTattempt'](*(batch_size,1))


#     # model = custom_models['simple'](*(8,1))
#     init_rngs =  {'params': main_key, 'dropout': dropout_key}
    
#     dummy_input = jax.random.randint(key, (batch_size, vector_length), minval=-1, maxval=20000)
    
#     # dummy_input = jax.random.randint(key, (vector_length), minval=-1, maxval=20000)

#     # inp = jax.random.normal(main_key, (batch_size,vector_length))

#     # params = simpleModel.init(key, np.ones([1,*model_size]))['params']
#     # params = model.init(init_rngs, jax.random.normal(init_rng, (batch_size,n_input)))['params']
    
#     variables = model.init(init_rngs, dummy_input)
#     # params = variables['params']
    
    
#     # TrainState is a simple built-in wrapper class that makes things a bit cleaner
#     # return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=opt), model
#     # use MyTrainState for batch_norm
#     return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'],  tx=opt), model


# Assuming `test_dataset.X['vector']` is a large array that you want to batch
def batched_apply(model, params, data, batch_size, dropout_rng):
    num_complete_batches, leftover = divmod(data.shape[0], batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_generator():
        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            yield data[start:end]
    i = 0
    all_logits = []
    for batch in data_generator():
        if len(batch)==batch_size:
          logits = model.apply({'params':params},batch,rngs={'dropout': dropout_rng})#.squeeze(axis=-1)
          
          all_logits.append(logits)#.squeeze(axis=-1))  # Adjust squeezing as necessary
    
    # Concatenate all batch results
    return jnp.concatenate(all_logits, axis=0)


def print_dict(dct):
  for item, values in dct.items():  # dct.iteritems() in Python 2
        print("{} ({})".format(item, values))


def boundary_filter(dataset):
    len_td = len(dataset.data.X)
    count=0

    for i in range(0,len_td):      
        idx = i - count
       
        if dataset.data.K['magnitude'][idx] > 0.45:   
            dataset.drop(idx)
            count+=1

    print(f'Dataset reduced from {len_td} to {len(dataset)} boundary points.')
    return dataset

jax.random.PRNGKey(42)

def inference(params,model, data):
    y = model.apply({'params': params}, data, train=False, rngs={'dropout': jax.random.PRNGKey(42)})
    # return y.squeeze(axis=-1)  
    return y


def predict(params, model, x, rng):
    y = model.apply({'params': params}, x, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y

batched_predict = jax.vmap(predict,in_axes=(None,None,0,None))
batched_inference = jax.vmap(inference,in_axes=(None,None,0))

def close_event():
            plt.close() #timer calls this function after 3 seconds and closes the window 


def plotEpoch(X, y, model, states,K = None, plot_type = None, name = 'untitled',project_dir = None,key = None):
    if key == None:
        key = jax.random.PRNGKey(42)
    
    if project_dir:
        pathname = project_dir
    else:
        pathname = os.getcwd()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    cm2 = plt.cm.PuOr

    normal_pal = sns.color_palette("Set1",(len(np.unique(y))+1)*2)
    pastel_pal = sns.color_palette("Pastel1",(len(np.unique(y))+1)*2)
    normal_pal.as_hex()
    pastel_pal.as_hex()
    
    for epoch,state in enumerate(states):
      
    #   model = custom_models[hyperparams['model']](*hyperparams['model_io'])
      Z,_ = model.apply({'params': state.params}, points)

      grad_map = jax.vmap(jax.grad(predict_wrapper, argnums=2), in_axes=(None, None, 0, None), out_axes=0)
      
      grads = grad_map(state.params, model,  points, key)
      magnitude = np.linalg.norm(grads, axis=1)

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
      timer = fig.canvas.new_timer(interval = 500) #creating a timer object and setting an interval of 3000 milliseconds
      timer.add_callback(close_event)

      
      
      im = ax1.contourf(xx, yy, magnitude.reshape(xx.shape), cmap=cm2, alpha=.8)
      fig.colorbar(im, ax=ax1)
      ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      
      if K:
            grad_fn = jax.grad(predict_wrapper_v2,argnums=1, allow_int=False)
    
            # Vectorize the gradient function over the batch of inputs using jax.vmap
            batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0, None),out_axes=1)
            
            # Now call the batched gradient function on the entire input array
            
            g_y = batched_grad_fn(state.params, X, key) * (2*jnp.array(y) - 1)[:,jnp.newaxis].T
            # g_y = batched_grad_fn(state.params, X, key) 
            EPS = 1e-8
            
            map_cosine = jax.vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                                    (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
            k_vector = jnp.multiply(jnp.array(K['vector']),jnp.array(K['magnitude']).reshape(-1,1,1))
            
            cosine_diff = jax.vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(k_vector)
            
            for i,gy in enumerate(g_y.T): 
            
                ax1.quiver(X[i,0],X[i,1],gy[0],gy[1],angles='xy', scale_units = 'xy',
                                          color=pastel_pal[y[i]],width=1/200,alpha=1.0,headlength=4,headwidth=4,scale=1)
                ax1.quiver(X[i,0],X[i,1],k_vector[i,0,0],k_vector[i,0,1],angles='xy', scale_units = 'xy',
                                          color=normal_pal[y[i]],width=1/200,alpha=1.0,headlength=4,headwidth=4,scale=1)
      ax1.set_xlim(lims[0])
      ax1.set_ylim(lims[1])

      im = ax2.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
      fig.colorbar(im, ax=ax2)
      
      ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      ax2.set_xlim(lims[0])
      ax2.set_ylim(lims[1])
      

      if plot_type == 'video':
        
        os.makedirs(pathname+"/video",exist_ok=True)
        os.makedirs(pathname+"/video/tmp",exist_ok=True)
        plt.savefig(pathname + "/video/tmp/file%02d.png" % epoch)  
        plt.close()
      else:  
        timer.start()
        plt.show()
        
    if plot_type == 'video':
      
      subprocess.call([
              'ffmpeg', '-framerate', '3','-loglevel', 'quiet', '-i',pathname + "/video/tmp/file%02d.png", '-r', '30', '-pix_fmt', 'yuv420p','-y',
              pathname + f"/video/{name}.mp4"])
      
      for file_name in glob.glob(pathname + "/video/tmp/*.png" ):
          os.remove(file_name)
    

def generate_figure(hyperparams, X, y, state):
    
    # plt.close('all')
    if plt.get_fignums():
      gui_fig = plt.figure(plt.get_fignums()[0])
      plt.figure(gui_fig.number)
      
    else:
       
       gui_fig = plt.figure()
    
    rect = gui_fig.patch
    rect.set_facecolor('lightslategray')

    ax = gui_fig.add_axes([0.1,0.1,0.75,0.8])
    cax = gui_fig.add_axes([0.85,0.1,
                            0.05,0.8])
    # ax.scatter([0,10],[0,10])
    # ax = gui_fig.add_subplot(1,1,1)
    # gui
    # cax = gui_fig.add_subplot(1,2,1)
    # gui_axes = [ax,cax]

    # ax= axes[0]
    # cax = axes[1]
    # gui_fig.clf()
    # ax.cla()
    # cax.cla()
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
  
    model = custom_models[hyperparams['model']](*hyperparams['model_size'])
    Z = model.apply({'params': state.params}, points)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
    # fig.colorbar(im, ax=ax)
    # ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k')
    # scatter = fig.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k',ax=ax)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    gui_fig.colorbar(im,cax=cax)
    gui_fig.show()

    return gui_fig, ax, cax
    


def generate_figure_gui(hyperparams, X, y, state):

    gui_fig = plt.figure()

    ax = gui_fig.add_axes([0.1,0.1,0.75,0.8])
    cax = gui_fig.add_axes([0.85,0.1,
                            0.05,0.8])
    # ax.scatter([0,10],[0,10])
    # ax = gui_fig.add_subplot(1,1,1)
    # gui
    # cax = gui_fig.add_subplot(1,2,1)
    # gui_axes = [ax,cax]

    # ax= axes[0]
    # cax = axes[1]
    # gui_fig.clf()
    # ax.cla()
    # cax.cla()
    
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
  
    model = custom_models[hyperparams['model']](*hyperparams['model_size'])
    Z = model.apply({'params': state.params}, points)

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    im = ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
    # fig.colorbar(im, ax=ax)
    # ax.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k')
    # scatter = fig.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,edgecolors='k',ax=ax)
    ax.set_xlim(lims[0])
    ax.set_ylim(lims[1])
    gui_fig.colorbar(im,cax=cax)
    
    # gui_fig.add_axes(ax)
    # gui_fig.add_axes(cax)

    return gui_fig, ax, scatter


def update_figure():
   None

def interactivePlot2():
   plt.scatter([0,1,2,903],[0,1,2,3])
   plt.show()



def gen_knowledge(dataset, knowledge_func):
    # dataset.data.K['vector'],dataset.data.K['label'],dataset.data.K['magnitude'] = (knowledge_func(dataset,n_vec=dataset.n_vec)) Need to correct old stuff from this
    dataset.data.K = knowledge_func(dataset.data.X,dataset.data.optimum_classifier,n_vec=dataset.n_vec) # to this

def save_stats(dict,name, path = os.getcwd()+'/results'):
  print(name)
  with open(name,"w") as fp:
    json.dump(dict,fp) 

def plot_stats(paths):

  fig,axs = plt.subplots(1)
  ax = axs
  ax.set_ylim((0,110))
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

  ax.set_ylabel('Accuracy (%)')
  ax.set_xlabel('Epochs')
  for path in paths:
    stats = json.load(path)
    ax.plot(range(0,len(stats['accuracy'])),stats['accuracy'],label="Counterfactual Vectors") #,alpha=1.0,marker='+',linestyle='dashed')
  plt.show()

def gen_savepath(data_params,hyperparams):
  dataset = str(data_params['dataset'])
  knowledge = str(data_params['knowledge_func'])
  loss = str(hyperparams['loss_function'])
  size = 'size_'+str(data_params['size'])
  epochs = 'epochs_'+str(hyperparams['epochs'])

  path_order = [dataset, knowledge, loss, size, epochs]

  savepath = os.getcwd()+"/results/" + "/".join(path_order) + '/'
  Path(savepath).mkdir(parents=True, exist_ok=True)
  
  return savepath


def assert_grads_equal(grads1, grads2, tol=1e-6):
    """Asserts that all leaves in two gradient PyTrees are equal within a tolerance."""
    def allclose(x, y):
        return jnp.allclose(x, y, rtol=tol, atol=tol)

    comparison = jax.tree_util.tree_map(allclose, grads1, grads2)
    assert jax.tree_util.tree_all(comparison), "Gradient dictionaries are not equal."
# @jax.jit  # Jit the function for efficiency
# @profile
def train_step(state, model, batch, loss_function, rng):
    # (_, logits), grads = jax.value_and_grad(loss_function, has_aux=True, argnums=0, allow_int=True)(state.params, model, batch, rng, config)
    _, grads = jax.value_and_grad(loss_function, has_aux=False, argnums=0)(state.params, model, batch, rng)
    
    if False:
        
        for path, grad in traverse_util.flatten_dict(grads).items():
            norm = jnp.linalg.norm(grad)
            print(" ".join(path), "NORM: ",norm,"MAX grad: ", jnp.max(jnp.array(grad)),"| MEAN grad: ", jnp.mean(jnp.array(grad)))
    
    state = state.apply_gradients(grads=grads)
    logits,_  = model.apply({'params': state.params}, np.array(batch['X']), train=False, rngs={'dropout': rng})
    metrics = compute_metrics(logits=logits, labels=np.array(batch['Y']))

    return state, metrics



# @jax.jit  # Jit the function for efficiency
def train_step_batch(state, model, batch, loss_function, rng):

    (_, (logits, batch_stats)), grads = jax.value_and_grad(loss_function, has_aux=True, argnums=0, allow_int=True)(state.params,state.batch_stats, model, batch, rng)
    # Check if gradients are all zero or not
    if False:
      for path, grad in traverse_util.flatten_dict(grads).items():
          norm = jnp.linalg.norm(grad)
          print(".".join(path), norm)
    state = state.apply_gradients(grads=grads)

    metrics = compute_metrics(logits=logits, labels=batch['Y'])

    return state, metrics


def train_one_epoch(state, dataset, model, loss_function, rng, visualise=False):

    batch_metrics = {'loss':[],'accuracy':[]}
    

    # from custom datsets - getitem: batch -> X, y, direction, direction label, direction distance 
    
    state, metrics = train_step(state, model, dataset, loss_function, rng)
    
    # batch_metrics.append(metrics)
    batch_metrics['loss'].append(metrics['loss'])
    batch_metrics['accuracy'].append(metrics['accuracy'])

    batch_metrics_np = jax.device_get(batch_metrics)  # pull from the accelerator onto host (CPU)
    
    epoch_metrics_np = {'loss':np.mean(batch_metrics_np['loss']),
                        'accuracy':np.mean(batch_metrics_np['accuracy']),
                        }
    
    return state, epoch_metrics_np


def gen_cf_vec(embedding):
   
   return

def embedding_knowledge(embeddings):
   for embedding in embeddings:
      if embedding['paired']:
         K = gen_cf_vec(embedding['embedding'])
      else:
         K = {'vector': [[] for _ in range(np.shape(embedding['embedding'])[0])]}
    
   return K

def get_max_dimension_and_index(jagged_lists):
    max_dimension = 0
    max_index = 0

    for i, lst in enumerate(jagged_lists):
        dimension = len(lst)
        if dimension > max_dimension:
            max_dimension = dimension
            max_index = i

    return max_dimension, max_index

def jagged_lists_to_array(jagged_lists):
    max_l, max_i = get_max_dimension_and_index(jagged_lists)
    
    try:
      list_dim = len(jagged_lists[max_i][0]) 
    except:
       list_dim=0
    result = []

    for lst in jagged_lists:    # pads each row with nans
        padded_lst = lst + [[np.nan]*list_dim] * (max_l - len(lst))
        result.append(padded_lst)
    
    return np.array(result)

def convert_to_list_of_lists(data):
    result = []
    for item in data:
        if isinstance(item, list):
            result.append(item)
        elif isinstance(item, np.ndarray):
            result.append(item.tolist())
    return result

def create_identical_matrix(array):
    # nan_mask = jnp.isnan(array)
    # identical_matrix = jnp.where(nan_mask, 0, 1)
    # return identical_matrix
    return jnp.where(jnp.isnan(array), 0, 1)


import flax.linen as nn
import jax.numpy as jnp

class Classifier(nn.Module):
    output_dim: int = 1  # binary classification output
    
    @nn.compact
    def __call__(self, avg_embed):
        # avg_embed: [batch_size, embed_dim]
        logits = nn.Dense(self.output_dim)(avg_embed)  # linear layer
        logits = logits.squeeze(-1)  # [batch_size]
        return logits

class EmbeddingOnlyModel(nn.Module):
    @nn.compact
    def __call__(self, embedded_inputs):  # embedded_inputs: (batch, embed_dim)
        logits = nn.Dense(features=1, name='linear1')(embedded_inputs)

        return nn.sigmoid(logits).squeeze(axis=-1)

def classifier_apply(params, avg_embed):
    model = EmbeddingOnlyModel()
    return model.apply({'params': params}, avg_embed)

def predict_wrapper_embedding_archive(params,embedding):
        linear_params = {'linear1': params}
        # Apply classifier part of model on averaged embedding
        return classifier_apply(linear_params, embedding) #.squeeze(-1)

embedding_only = EmbeddingOnlyModel()

# def predict_wrapper_embedding(linear_params,embedding):
#         return embedding_only.apply({'params': linear_params}, embedding)
        
testModel = cm.SimpleClassifier_v2(8,1)

def predict_wrapper(params, model, x, rng):
    y,_ = model.apply({'params': params}, jnp.array([x]), train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y.squeeze(axis=-1)


def predict_wrapper2(params, model, x, rng):
    y,_ = model.apply({'params': params}, jnp.array([x]), train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y

def predict_wrapper_v2(params, x, rng):
    y,_ = testModel.apply({'params': params}, jnp.array([x]), train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y.squeeze(axis=-1)


def predict_wrapper_embedding(linear_params,embedding):
        return embedding_only.apply({'params': linear_params}, embedding)
        

# def masked_average_embeddings(embedded_tokens, tokens, padding_idx=0):
#     print(np.shape(embedded_tokens), np.shape(tokens))
#     mask = (tokens != padding_idx).astype(jnp.float32)  # (batch, seq_len)
#     mask = jnp.expand_dims(mask, axis=-1)              # (batch, seq_len, 1)
    
#     masked_embeds = embedded_tokens * mask              # zero out padding embeddings
    
#     sum_embeds = jnp.sum(masked_embeds, axis=1)         # sum over seq_len
#     lengths = jnp.sum(mask, axis=1) + 1e-8               # lengths per sample
    
#     avg_embeds = sum_embeds / lengths                    # average embeddings
    
#     return avg_embeds  # (batch, embed_dim)
def masked_average_embeddings(embedded_tokens, tokens, padding_idx=-1):
    # embedded_tokens: (batch_size, seq_len, embed_dim)
    # tokens: (batch_size, seq_len)
    
    mask = (tokens != padding_idx).astype(jnp.float32)  # (batch_size, seq_len)
    mask = mask[..., None]  # expand dims to (batch_size, seq_len, 1) for broadcasting

    masked_embeddings = embedded_tokens * mask  # zero-out embeddings of padding tokens
    summed = jnp.sum(masked_embeddings, axis=1)  # sum over seq_len -> (batch_size, embed_dim)
    counts = jnp.sum(mask, axis=1)  # count non-padding tokens (batch_size, 1)

    # Avoid division by zero
    counts = jnp.maximum(counts, 1.0)

    avg_embed = summed / counts  # (batch_size, embed_dim)
    return avg_embed


def embed_and_average(params, tokens):
    tokens = jnp.array(tokens)  # Ensure tokens is a JAX array
    # params['embed']['embedding'] is your embedding matrix of shape (vocab_size, embed_dim)
    embeddings = params['embed']['embedding']  
    # embedded_tokens = embeddings[tokens]  # (batch, seq_len, embed_dim)
    embedded_tokens = jnp.take(embeddings, tokens, axis=0)
    
    avg_embed = masked_average_embeddings(embedded_tokens, tokens, padding_idx=-1)
    return avg_embed

def embed_and_average_batchK(params, K_vectors):
    # K_vectors: (batch, num_K, seq_len)
    batch_size, num_K, seq_len = K_vectors.shape

    # Flatten batch and num_K dimensions for embedding
    K_flat = K_vectors.reshape(batch_size * num_K, seq_len)  # (batch*num_K, seq_len)

    # Embed and average using your existing function
    avg_embed_flat = embed_and_average(params, K_flat)  # (batch*num_K, embed_dim)

    # Reshape back to (batch, num_K, embed_dim)
    avg_embed = avg_embed_flat.reshape(batch_size, num_K, -1)  # (batch, num_K, embed_dim)

    return avg_embed

# def cosine_similarity_batch(X, Y, eps=1e-8):
#         # g_y: (batch, embed_dim)
#         # K_avg: (batch, num_K, embed_dim)
#         g_y_expanded = jnp.expand_dims(X, axis=1)  # (batch, 1, embed_dim)
        
#         dot = jnp.sum(g_y_expanded * K_avg, axis=-1)  # (batch, num_K)
#         norm_g_y = jnp.linalg.norm(g_y_expanded, axis=-1)  # (batch, 1)
#         norm_K = jnp.linalg.norm(K_avg, axis=-1)  # (batch, num_K)
        
#         cos_sim = 1 - dot / (norm_g_y * norm_K + eps)  # cosine distance (lower = more similar)
#         return cos_sim  # (batch, num_K)

def cosine_distance_batch(X, K, eps=1e-8):
    """
    Compute cosine similarity between each X[i] and all K[i, j].

    Args:
        X: (batch, dim)
        K: (batch, max_k, dim)
        eps: small constant to avoid division by zero

    Returns:
        cos_sim: (batch, max_k)
    """
    # Normalize along the last axis (vector dimension)
    X_norm = X / (jnp.linalg.norm(X, axis=-1, keepdims=True) + eps)           # (batch, dim)
    K_norm = K / (jnp.linalg.norm(K, axis=-1, keepdims=True) + eps)           # (batch, max_k, dim)

    # Expand X so it can broadcast against K
    X_expanded = X_norm[:, None, :]                                           # (batch, 1, dim)

    # Cosine similarity is just the dot product after normalization
    cos_sim = jnp.sum(X_expanded * K_norm, axis=-1)                           # (batch, max_k)

    return 1 - cos_sim # cosine distance
