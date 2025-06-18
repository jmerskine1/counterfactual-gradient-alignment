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


def combine_datasets(data1,data2):
    
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

def custom_collate(batch):
    
    x_vec = [b['X']['vector'] for b in batch]
    x_txt = [b['X']['text'] for b in batch]
    labels = [b['Y'] for b in batch]
    k_txt = [b['K']['text'] for b in batch]
    k_vec = [b['K']['vector'] for b in batch]
    k_lab = [b['K']['label'] for b in batch]
    k_mag = [b['K']['magnitude'] for b in batch]
    
    x_vec = jnp.stack(x_vec)
    X = {'vector':x_vec,'text':x_txt}
    labels = jnp.stack(labels)
    # knowledge = {'text':k_txt,'vector':jnp.stack(k_vec),'label':k_lab,'magnitude':k_mag}
    knowledge = {'text':k_txt,'vector':jnp.stack([jnp.stack(k_v) for k_v in k_vec]),'label':k_lab,'magnitude':k_mag}

    return{'X':X,'Y':labels,'K':knowledge}

def custom_collate_2D(batch):
    
    x_vec = [b['X']['vector'] for b in batch]
    
    labels = [b['Y'] for b in batch]
    
    k_vec = [b['K']['vector'] for b in batch]
    k_lab = [b['K']['label'] for b in batch]
    k_mag = [b['K']['magnitude'] for b in batch]
    
    x_vec = jnp.stack(x_vec)
    X = {'vector':x_vec}
    labels = jnp.stack(labels)
    # knowledge = {'text':k_txt,'vector':jnp.stack(k_vec),'label':k_lab,'magnitude':k_mag}
    knowledge = {'vector':jnp.stack([jnp.stack(k_v) for k_v in k_vec]),'label':k_lab,'magnitude':k_mag}

    return{'X':X,'Y':labels,'K':knowledge}

# def custom_collate(batch):
#     x_vec = [b['X']['vector'] for b in batch]
#     x_txt = [b['X']['text'] for b in batch]
#     labels = [b['Y'] for b in batch]
#     k_txt = [b['K']['text'] for b in batch]
#     k_vec = [b['K']['vector'] for b in batch]
#     k_lab = [b['K']['label'] for b in batch]
#     k_mag = [b['K']['magnitude'] for b in batch]

#     # Print statements for debugging
#     print('X:', len(x_vec[0]), 'Shape:', jnp.array(x_vec[0]).shape)
#     print('K:', len(k_vec[0]), 'Shape:', jnp.array(k_vec[0]).shape)

#     for k in k_vec:
#        for cf in k:
#           if any(math.isnan(x) for x in cf):
#              print(cf)

#     # Stack X['vector'] and Y as before
#     x_vec = jnp.stack(x_vec)
#     labels = jnp.stack(labels)

#     # Handle stacking of K['vector'] for 3D arrays
#     # If K['vector'] is 3D, stack directly along the first axis
#     if len(k_vec[0].shape) == 3:
#         k_vec_stacked = jnp.stack(k_vec)
#     else:
#         # Fallback in case it's not 3D, handle 2D or other cases
#         k_vec_stacked = jnp.stack([jnp.stack(k_v) for k_v in k_vec])

#     X = {'vector': x_vec, 'text': x_txt}
#     knowledge = {'text': k_txt, 'vector': k_vec_stacked, 'label': k_lab, 'magnitude': k_mag}

#     return {'X': X, 'Y': labels, 'K': knowledge}

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



c_palette = sns.palettes.color_palette()
cmapList= list(colormaps)

def plot_from_results_file(results, xlims = None, ylims = None, loss = False, labels=None):
        if type(results) != list:
                results = [results]
        
        cmaps = list(colormaps)
        # Create figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111)

        
        
        
        
        linestyles = [ ':','-.', '--','-']

        for i,results in enumerate(results):
                col = c_palette[i]
                # Create figure and 3D axis
                
                count = 0
                for name, res in results.items():
                        
                        if labels:
                                if i < len(labels):
                                        name = f'{labels[i]} - {name}'


                        accuracy = res['accuracy']
                        if len(accuracy)==0:
                                continue
                        
                        
                        x = range(len(accuracy))
                        line = ax.plot(x,accuracy,linestyle=linestyles[count],label=name, color=col)
                        
                        # Find the maximum accuracy and its corresponding x-value
                        max_acc = np.max(accuracy)
                        max_idx = np.argmax(accuracy)

                        # Plot a marker at the maximum accuracy point using the line's color
                        ax.plot(max_idx, max_acc, 'o', color=col, markersize=8)

                        # Add a label next to the marker, color coded to match the line color
                        ax.text(1, max_acc, f'{max_acc*100:.1f}%', fontsize=10, 
                                verticalalignment='bottom', horizontalalignment='right', 
                                color=col)
                        
                        if 'train' in name.lower():
                                if loss:
                                        loss = res['losses']
                                        x = range(len(loss))
                                        # ax.plot(x,loss,linestyle=linestyles[count], color=col,alpha=0.)
                                        ax.scatter(x,loss,marker='+', s=20, color=col,alpha=0.5)
                                        
                                
                        count+=1


        

        # Add legend and labels
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        
        if xlims:
                ax.set_xlim(xlims)
        
        if ylims:
                ax.set_ylim(ylims)

        plt.show()
        return fig,ax


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

def generate_results(data,model,params,name='Untitled'):
        X = data[0]
        Y = data[1]
        logits = model.apply({'params':params},np.array(X))
        metrics = compute_metrics(logits,np.array(Y))
        print(f"{name} Loss: {metrics['loss']}, {name} Accuracy: {metrics['accuracy'] * 100}")
        return metrics

def generate_results_ensemble(X,Y,models,params,name='Untitled'):
        logits = np.zeros((len(models),len(X['vector'])))
        
        for i, (model, param) in enumerate(list(zip(models,params))):
            logits[i,:] = model.apply({'params':param},np.array(X['vector']),train=False)
        
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
    dummy_input = jax.random.normal(main_key, (1, vector_length))
    
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

def predict_wrapper(params, model, x, rng):
    
    y = model.apply({'params': params}, jnp.array([x]), train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y.squeeze(axis=-1)

def predict(params, model, x, rng):
    y = model.apply({'params': params}, x, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y

batched_predict = jax.vmap(predict,in_axes=(None,None,0,None))
batched_inference = jax.vmap(inference,in_axes=(None,None,0))

def close_event():
            plt.close() #timer calls this function after 3 seconds and closes the window 


def plotEpoch(X, y, model, states, plot_type = None, name = 'untitled',key = None):
    if key == None:
        key = jax.random.PRNGKey(42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    lims = [[x_min, x_max], [y_min, y_max]]
    xx, yy = np.meshgrid(np.arange(lims[0][0], lims[0][1], 0.01),
                            np.arange(lims[1][0], lims[1][1], 0.01))
    points = np.stack([xx.ravel(), yy.ravel()]).T
    
    for epoch,state in enumerate(states):
      
    #   model = custom_models[hyperparams['model']](*hyperparams['model_io'])
      Z = model.apply({'params': state.params}, points)

      grad_map = jax.vmap(jax.grad(predict_wrapper, argnums=2), in_axes=(None, None, 0, None), out_axes=0)
      
      grads = grad_map(state.params, model,  points, key)
      magnitude = np.linalg.norm(grads, axis=1)

      fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharey='row')
      timer = fig.canvas.new_timer(interval = 500) #creating a timer object and setting an interval of 3000 milliseconds
      timer.add_callback(close_event)

      cm = plt.cm.RdBu
      cm_bright = ListedColormap(['#FF0000', '#0000FF'])
      cm2 = plt.cm.PuOr
      
      im = ax1.contourf(xx, yy, magnitude.reshape(xx.shape), cmap=cm2, alpha=.8)
      fig.colorbar(im, ax=ax1)
      ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      ax1.set_xlim(lims[0])
      ax1.set_ylim(lims[1])

      im = ax2.contourf(xx, yy, Z.reshape(xx.shape), cmap=cm, alpha=.8)    
      fig.colorbar(im, ax=ax2)
      
      ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright,
                      edgecolors='k')
      ax2.set_xlim(lims[0])
      ax2.set_ylim(lims[1])
      

      if plot_type == 'video':
        os.makedirs("video",exist_ok=True)
        os.makedirs("video/tmp",exist_ok=True)
        plt.savefig(os.getcwd() + "/video/tmp/file%02d.png" % epoch)  
        plt.close()
      else:  
        timer.start()
        plt.show()
        
    if plot_type == 'video':
      
      subprocess.call([
              'ffmpeg', '-framerate', '3','-loglevel', 'quiet', '-i',os.getcwd() + "/video/tmp/file%02d.png", '-r', '30', '-pix_fmt', 'yuv420p','-y',
              os.getcwd() + f"/video/{name}.mp4"])
      
      for file_name in glob.glob(os.getcwd() + "/video/tmp/*.png" ):
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
    dataset.data.K = knowledge_func(dataset.data.X['vector'],dataset.data.optimum_classifier,n_vec=dataset.n_vec) # to this

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


# @jax.jit  # Jit the function for efficiency
# @profile
def train_step(state, model, batch, loss_function, rng, config):
    
    (_, logits), grads = jax.value_and_grad(loss_function, has_aux=True, argnums=0, allow_int=True)(state.params, model, batch, rng, config)
    
    if False:
      for path, grad in traverse_util.flatten_dict(grads).items():
          norm = jnp.linalg.norm(grad)
          print(".".join(path), norm)
    
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=np.array(batch[1]))

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


def train_one_epoch(state, data_loader, model, loss_function, rng, config, visualise=False):

    batch_metrics = {'loss':[],'accuracy':[]}
    
    for batch in data_loader:
        # from custom datsets - getitem: batch -> X, y, direction, direction label, direction distance 
        
        state, metrics = train_step(state, model, batch, loss_function, rng, config)
        
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


