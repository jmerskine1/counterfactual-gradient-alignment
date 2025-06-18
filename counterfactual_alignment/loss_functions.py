

from counterfactual_alignment.utilities import (predict_wrapper, 
                                                           get_unit_vec,
                                                           get_max_dimension_and_index, 
                                                           jagged_lists_to_array,
                                                           convert_to_list_of_lists, 
                                                           create_identical_matrix)
from counterfactual_alignment.custom_models import custom_models
import numpy as np
import sys
import jax.numpy as jnp
import jax
from jax import grad, vmap

import torch
import optax
import yaml
import os



def cross_entropy_batch(params, batch_stats, model, batch, rng):
    X,Y,K = batch['X'],batch['Y'],batch['K']
    # print(X['vector'][0],Y[0],'\n')
    # model = custom_models['GPTattempt'](*(8,1))
    # model = custom_models['GPTattempt']()
    
    logits, batch_stats = model.apply({'params': params, 'batch_stats':batch_stats}, X['vector'], train=True, rngs={'dropout': rng}, mutable=['batch_stats']) #.squeeze(axis=-1)  
    
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    # print([(label,logit.primal) for label,logit in list(zip(Y,logits))])
    
    return loss, (logits, batch_stats)

def predict(params, model, x, rng):
    y = model.apply({'params': params}, x, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  
    return y

batched_predict = jax.vmap(predict,in_axes=(None,None,0,None))

# @profile
def cross_entropy(params, model, batch, rng, config):
    X,Y,K = batch[0],batch[1],batch[2]
    logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=np.array(Y)))
    
    return loss, logits


# @profile
def cross_entropy_archive(params, model, batch, rng):
    X,Y,K = batch['X'],batch['Y'],batch['K']
    # print(X['vector'][0],Y[0],'\n')
    # model = custom_models['GPTattempt'](*(8,1))
    # model = custom_models['GPTattempt']()
    
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    # print([(label,logit.primal) for label,logit in list(zip(Y,logits))])
    
    return loss, logits

# @profile
def gradient_supervision_archive_2(params, model, batch, rng):
    X,Y,K = batch['X'],batch['Y'],batch['K']
    # print(X['vector'][0],Y[0],'\n')
    # model = custom_models['GPTattempt'](*(8,1))
    # model = custom_models['GPTattempt']()
    
    # logits, inters = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}, capture_intermediates=True) #.squeeze(axis=-1)  
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    # print([(label,logit.primal) for label,logit in list(zip(Y,logits))])

    loss_gs = []

    α = config['hyperparams']['loss_mix'] 
    
    # get embedding from x
    # get embedding from k
    # g_x = k - x
    
    
    # x_vec = np.mean(inters['intermediates']['Embed_0']['__call__'][0].primal, axis=1) 
    # print(x_vec.shape)
    # print(X['vector'].shape)
    # print(K['vector'].shape)
    # k_logits, k_inters = model.apply({'params': params}, K['vector'], train=False, rngs={'dropout': rng}, capture_intermediates=True) #.squeeze(axis=-1)  
    # k_vec = np.mean(k_inters['intermediates']['Embed_0']['__call__'][0].primal, axis=1) 
    # print(k_vec.shape)
    for i,d in enumerate(X['vector']):
        # g_y = grad(predict_wrapper,argnums=2)(model, params, d[np.newaxis,:], rng).squeeze(axis=0)  
        g_y = grad(predict_wrapper,argnums=2)(model, params, d, rng)
        # logits = model.apply({'params': params}, d, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  

        for j,g_x in enumerate(K['vector'][i]):
            if np.sum(g_x)==0:
                continue
            else:
                # print("K",k)
            # need to get the embedding out...
                # print(K['magnitude'][i][j])
                # g_x = k # * K['magnitude'][i][j]
                # print("G_X",g_x)
                # make unit vectors, inspect
                # loss_i = 1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)+1e-5)
                # track cross product over time
                # g_y=g_y[jnp.newaxis,:]
                # g_x=g_x[jnp.newaxis,:]

                # print('dc',np.shape(g_x),np.shape(g_y))
                # dot = jnp.dot(g_x,g_y)
                # cross = jnp.linalg.norm(g_x)*jnp.linalg.norm(g_y)
                # cross = jnp.cross(g_x,g_y)
                
                gs_loss = 1 - (jnp.dot(g_x,g_y)/
                               (jnp.linalg.norm(g_x)*jnp.linalg.norm(g_y)))

                # print(f'Dot: {dot.primal}\nCross: {cross.primal}')
                # loss_i = jnp.where(jnp.isnan(loss_i),0,loss_i)
                # loss_gs = jnp.append(loss_gs,1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)))
                
                loss_gs.append(gs_loss)
                
                
    
    loss = loss + α*jnp.mean(jnp.array(loss_gs))
    
    return loss, logits

"""
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
        # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)

    # Now call the batched gradient function on the entire input array
    g = batched_grad_fn(params, model, X['vector'], rng)
    
    # The following could be made less comp.icated if we assume only counterfactual per example - as it is, it allows for K to be 2D
    map_dd = vmap(lambda a_row, b_col: jnp.dot(a_row, b_col), in_axes=(0, 1))
    directional_derivative = vmap(lambda K_slice: map_dd(K_slice, g), in_axes=1)(K['vector'])

    sign = jnp.tanh(1e5*directional_derivative)
    d_loss = jnp.mean(sign*Y*2 - sign + 1)
    loss = (1-α)*ce_loss + α*d_loss

    if False:
        jax.debug.print("CE_LOSS: {x}",x = ce_loss)
        jax.debug.print("D_LOSS: {x}",x = d_loss)
        # jax.debug.print("SIGN: {x}",x = sign)
        # jax.debug.print("Y: {x}",x = Y)
        # jax.debug.print("LOSS: {x}",x = sign*Y*2 - sign + 1)
    
    # loss = loss1
    return loss, logits
"""




def gradient_supervision_basic(params, model, batch, rng):
    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)
    
    # Now call the batched gradient function on the entire input array
    g_y = batched_grad_fn(params, model, X['vector'], rng)
    EPS = 1e-8
    alt_loss = []

    for i in range(len(X['vector'])):

        g_y_i = grad_fn(params, model, X['vector'][i], rng)
        g_hat = K['vector'][i][0]*-1

        cosine_sim_i = (jnp.dot(g_y_i, g_hat) /
                              (jnp.linalg.norm(g_y_i) * jnp.linalg.norm(g_hat) + EPS))
        loss_i = 1 - cosine_sim_i
        
        alt_loss.append(loss_i)
        
        if False:
            print("x:",X['vector'][i])
            print("y:",Y[i])
            jax.debug.print('g_y_i {}',g_y_i)
            jax.debug.print('g_hat {}',g_hat)        
            jax.debug.print('cosine similarity: {}',cosine_sim_i)
            jax.debug.print('loss_i {}',loss_i)

    alt_loss = jnp.mean(jnp.array(alt_loss))

    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))

    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(K['vector'])
    
    α = config['hyperparams']['loss_mix']
    gs_loss = jnp.mean(jnp.array(cosine_sim))
    
    if False:
        jax.debug.print('alt_loss {}',alt_loss)
        jax.debug.print('og loss {}',jnp.mean(jnp.array(cosine_sim)))
        
        jax.debug.print('cosim {}',cosine_sim)
        jax.debug.print('gs_loss {}',gs_loss)
        jax.debug.print('ce_loss {}',ce_loss)

    # loss = ce_loss + gs_loss
    loss = alt_loss

    # jax.debug.print("Gradient Loss: {x}",x=gs_loss)
    
    return loss, logits


def gradient_supervision(params, model, batch, rng):
    
    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    # Y = Y-1
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)
    
    # Now call the batched gradient function on the entire input array
    g_y = batched_grad_fn(params, model, X['vector'], rng)
    EPS = 1e-8
    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))

    # The following could be made less complicated if we assume only counterfactual per example - as it is, it allows for K to be 2D
    # map_cosine = vmap(lambda a_row, b_col:  1 - (jnp.dot(a_row,b_col)/
    #                            (jnp.linalg.norm(a_row)*jnp.linalg.norm(b_col))), in_axes=(0, 1))
    

    # test = map_cosine(g_y,K['vector'][:,0,:])
    # print(test[0].primal)
    # print("TEST: ",np.shape(test))
    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(K['vector'])
    
    
    # print("COSINE SIM SHAPE", np.shape(cosine_sim))
    # if not jnp.allclose(cosim,cosine_sim):
    
    #     print("COSIM Failed")
    #     sys.exit(0)

    α = config['hyperparams']['loss_mix']
    
    gs_loss = α*jnp.mean(jnp.array(cosine_sim))
    
    if False:
        jax.debug.print('g_y {}',g_y)
        jax.debug.print('cosim {}',cosine_sim)
        jax.debug.print('gs_loss {}',gs_loss)
        jax.debug.print('ce_loss {}',ce_loss)

    loss = ce_loss + gs_loss

    # jax.debug.print("Gradient Loss: {x}",x=gs_loss)
    
    return loss, logits

def l2_loss(x, alpha):
    return alpha * (x ** 2).mean()

def cross_entropy_l2(params, batch):
    X,Y,K = batch['X'],batch['Y'],batch['K']


    key = jax.random.PRNGKey(42)
    model = custom_models['simple'](*(8,1))

    logits = model.apply({'params': params}, X).squeeze(axis=-1)  
    loss = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    lambda_reg = 0.002
    loss = loss + sum(l2_loss(w, alpha=lambda_reg) for w in jax.tree_util.tree_leaves(params))

    return loss, logits


def direction(params, model, batch, rng):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    # logits, inters = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}, capture_intermediates=True) #.squeeze(axis=-1)  
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))

    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
        # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)

    # Now call the batched gradient function on the entire input array
    g = batched_grad_fn(params, model, X['vector'], rng)
    
    # The following could be made less comp.icated if we assume only counterfactual per example - as it is, it allows for K to be 2D
    map_dd = vmap(lambda a_row, b_col: jnp.dot(a_row, b_col), in_axes=(0, 1))
    
    directional_derivative = vmap(lambda K_slice: map_dd(K_slice, g), in_axes=1)(K['vector'])
    
    # print("dd: ",np.shape(directional_derivative))
    # print("K: ",np.shape(K['vector']))
    # jax.debug.print("dd: {x}",x=directional_derivative)
    sign = jnp.tanh(1e5*directional_derivative)
    d_loss = jnp.mean(sign*Y*2 - sign + 1)

    # jax.debug.print("Y: {x}",x=Y)
    # jax.debug.print("sign: {x}",x=sign)
    # jax.debug.print("d_loss: {x}",x=d_loss)

    α = config['hyperparams']['loss_mix']
    loss = (1-α)*ce_loss + α*d_loss

    if False:
        jax.debug.print("CE_LOSS: {x}",x = ce_loss)
        jax.debug.print("D_LOSS: {x}",x = d_loss)
        # jax.debug.print("SIGN: {x}",x = sign)
        # jax.debug.print("Y: {x}",x = Y)
        # jax.debug.print("LOSS: {x}",x = sign*Y*2 - sign + 1)
    
    # loss = loss1
    return d_loss, logits

# def direction_new(params, model, batch, rng):
#     X = jnp.array(batch[0])  # shape (N, ...)
#     Y = jnp.array(batch[1])  # shape (N,)
#     K = jnp.array(batch[2])  # shape (N, 2)

#     # Identify original examples
#     k_mask = (K[:, 0] == 0) & (K[:, 1] == 0)  # True for original points

#     # Forward pass over ALL data
#     logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})  # shape (N,)

#     # Compute CE loss only on original examples
#     if jnp.any(k_mask):
#         ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(
#             logits=logits[k_mask],
#             labels=Y[k_mask]
#         ))
#     else:
#         ce_loss = 0.0  # or raise error

#     α = config['hyperparams']['loss_mix']

#     # Compute directional loss only if α > 0
#     if α > 0:
#         # Compute gradient of model output wrt input
#         grad_fn = grad(predict_wrapper, argnums=2, allow_int=False)
#         batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None), out_axes=1)
#         g = batched_grad_fn(params, model, X, rng)  # shape (D, N)

#         # Expand K to (N, 1, D) for broadcasting with grad
#         K_exp = jnp.expand_dims(K, 1)  # shape (N, 1, D)

#         # Dot product between K and grads
#         directional_derivative = jnp.einsum('nid,idn->n', K_exp, g)

#         # Directional loss — only for counterfactuals
#         sign = jnp.sign(directional_derivative)
#         d_loss = sign * Y * 2 - sign + 1
#         d_loss = d_loss * (~k_mask)  # mask out original examples
#         d_loss = jnp.mean(d_loss)
#     else:
#         d_loss = 0.0
#     ce_loss = (1 - α) * ce_loss
#     print("ALPHA",α)
#     jax.debug.print("CE_LOSS: {x}",x = ce_loss)
#     loss = (1 - α) * ce_loss + α * d_loss

#     return loss, logits
def loss_wrapper(params, model, batch, rng, config):
    loss_terms = []
    logits = []
    losses = config['hyperparams']['loss_function']
    losses = [cross_entropy]
    if not isinstance(losses,list):
        losses = [losses]

    for loss_fn in losses:
        loss_i,logits_i = loss_fn(params, model, batch, rng, config)
        loss_terms.append(loss_i)
        logits.append(logits_i)

    return jnp.mean(jnp.stack(loss_terms)), jnp.stack(logits)

def direction_new_aklt(params, model, batch, rng, config):
    X = np.array(batch[0])  # shape (160, ...)
    Y = np.array(batch[1])  # shape (160,)
    K = np.array(batch[2])  # shape (160, 2)

    # Identify original data points (not counterfactuals)
    k_mask = (K[:, 0] == 0) & (K[:, 1] == 0)  # shape (160,) — True for original 40

    # Single forward pass
    logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})  # shape (160, ...)
    
    # Compute CE loss only for original examples
    if jnp.sum(k_mask) > 0:
        ce_logits = logits[k_mask]
        ce_labels = Y[k_mask]
        ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=ce_logits, labels=ce_labels))
    else:
        ce_loss = 0.0


    # Compute directional derivative loss (only matters if α > 0)
    if config['hyperparams']['loss_mix'] > 0:
        grad_fn = grad(predict_wrapper, argnums=2, allow_int=False)
        batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None), out_axes=1)
        g = batched_grad_fn(params, model, X, rng)

        map_dd = vmap(lambda a_row, b_col: jnp.dot(a_row, b_col), in_axes=(0, 1))
        K_exp = np.expand_dims(K, 1)
        directional_derivative = vmap(lambda K_slice: map_dd(K_slice, g), in_axes=1)(K_exp)

        sign = jnp.sign(directional_derivative)
        d_loss = sign * Y * 2 - sign + 1
        d_loss = d_loss * (~k_mask)
        d_loss = jnp.mean(d_loss)
    else:
        d_loss = 0.0

    # Combine
    α = config['hyperparams']['loss_mix']
    ce_loss = (1 - α) * ce_loss
    jax.debug.print("CE_LOSS: {x}",x = ce_loss)
    print("ALPHA",α)    
    loss = ce_loss + α * d_loss

    # Return logits for original examples (to match `cross_entropy`)
    return loss, logits


def direction_new(params, model, batch, rng, config):

    X,Y,K = np.array(batch[0]),np.array(batch[1]),np.array(batch[2])
    
    k_mask  = (K[:, 0] == 0) & (K[:, 1] == 0)
    CE_X = X[k_mask]
    CE_Y = Y[k_mask]
    
    logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    ce_logits = logits[k_mask]
    # ce_logits = model.apply({'params': params}, CE_X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    x_inds = np.nonzero(k_mask)

    ce_loss = optax.sigmoid_binary_cross_entropy(logits=ce_logits, labels=CE_Y)
    # jax.debug.print("ce_loss: {x}",x=ce_loss)
    # ce_loss = ce_loss[x_inds]
    # jax.debug.print("ce_loss: {x}",x=ce_loss)
    ce_loss = jnp.mean(ce_loss)

    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
        # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)

    # Now call the batched gradient function on the entire input array
    g = batched_grad_fn(params, model, X, rng)
    
    # The following could be made less comp.icated if we assume only counterfactual per example - as it is, it allows for K to be 2D
    map_dd = vmap(lambda a_row, b_col: jnp.dot(a_row, b_col), in_axes=(0, 1))
    K = np.expand_dims(K,1)

    directional_derivative = vmap(lambda K_slice: map_dd(K_slice, g), in_axes=1)(K)

    # jax.debug.print("directional_deriv: {x}",x=directional_derivative)
    # sign = jnp.tanh(1e5*directional_derivative)
    sign = jnp.sign(directional_derivative)
    # jax.debug.print("sign: {x}",x=sign)
    d_loss = sign*Y*2 - sign + 1
    # jax.debug.print("d_loss: {x}",x=d_loss)
    
    not_k_mask = np.array([not b for b in k_mask]) # d_loss[x_inds] = 0
    
    d_loss = d_loss * not_k_mask
    
    d_loss = jnp.mean(d_loss)

    # jax.debug.print("d_loss: {x}",x=d_loss)
    # jax.debug.print("ce_loss: {x}",x=ce_loss)
    # jax.debug.print("Y: {x}",x=Y)
    # jax.debug.print("sign: {x}",x=sign)
    

    α = config['hyperparams']['loss_mix']
    ce_loss = (1 - α) * ce_loss
    jax.debug.print("CE_LOSS: {x}",x = ce_loss)
    print("ALPHA",α)    
    loss = ce_loss + α * d_loss
    # jax.debug.print("loss: {x}",x=loss)
    if False:
        print('KMNASK',k_mask)
        print("XINDS",x_inds)
        print('sign: ',sign)
        print('dloss: ',sign*Y*2 - sign + 1)
        print("\nK: ",np.shape(K),K)
        print("\nG: ",np.shape(g),g)
        jax.debug.print("dd: {x}",x=np.shape(directional_derivative))
        jax.debug.print("CE_LOSS: {x}",x = ce_loss)
        jax.debug.print("D_LOSS: {x}",x = d_loss)
        # jax.debug.print("SIGN: {x}",x = sign)
        # jax.debug.print("Y: {x}",x = Y)
        # jax.debug.print("LOSS: {x}",x = sign*Y*2 - sign + 1)
    
    
    return loss, logits



def direction_archive(params, model, batch, rng):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    # logits, inters = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}, capture_intermediates=True) #.squeeze(axis=-1)  
    logits = model.apply({'params': params}, X['vector'], train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    d_loss = jnp.empty((0,1))

    α = 0.5

    for i,d in enumerate(X['vector']):
        g = grad(predict_wrapper,argnums=2)(model, params, d[np.newaxis,:], rng).squeeze(axis=0)  
        # g = grad(predict_wrapper,argnums=2)(model, params, d)

        print("KVEC",K['vector'][i])
        directional_derivative = g @ K['vector'][i].T 

        sign = jnp.tanh(200.0*directional_derivative)
        print("sign_pre: ",sign.primal)
        sign = sign*Y[i]*2 - sign
        print("sign: ",sign.primal,
              " | Label: ",Y[i]," | Loss term: ",jnp.abs(sign-K['label'][i]).primal)
        d_loss = jnp.append(d_loss,jnp.abs(sign-K['label'][i]))

    loss = (1-α)*loss + α*jnp.mean(d_loss)
    # loss = loss1
    return loss, logits

# @jax.jit
def direction_interactive_vectorized( params, batch):

    model = custom_models['simple'](*(8,1))

    loss_mix = 0.25
    X, Y, K = batch['X'], batch['Y'], batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)
    loss1 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    # loss2 = jnp.empty((0,))

    Kvec = jnp.array(K['vector'])
    count = jnp.count_nonzero(~jnp.isnan(Kvec))

    if np.shape(Kvec)[1] == 0 or jnp.any(~jnp.isnan(Kvec))==False:
        loss2 = 0
    else:
        
        g = jax.vmap(jax.grad(predict_wrapper,argnums=2),(None,None,0))(model, params, X)
        # g_reshaped = g[jnp.newaxis,:,:]

        Kmask = create_identical_matrix(Kvec)
        # count = jnp.count_nonzero(~jnp.isnan(Kvec))

        # K_nonan = jnp.nan_to_num(Kvec,nan=0)
        K_nonan = jnp.array(jnp.where(jnp.isnan(Kvec), 0.0, Kvec))
        
        # print("KONAN: ",jax.tree_util.tree_leaves(jnp.count_nonzero(jnp.isnan(K_nonan))))
        # directional_derivative = jnp.multiply(g[jnp.newaxis,:,:], K_nonan)
        directional_derivative = jnp.multiply(g[:,jnp.newaxis,:], K_nonan)
        
        # directional_derivative = custom_multiply(g_reshaped, K_nonan)
        
        sign = jnp.tanh(20.0 * directional_derivative)

        

        Y = jnp.array(Y)[:, jnp.newaxis, jnp.newaxis]
        
        sign = sign * Y * 2 - sign
        
        loss2 = jnp.sum(jnp.abs(sign+1)*Kmask)/count
    

    loss = (1 - loss_mix) * loss1 + loss_mix * loss2
    
    return loss, logits


# @jax.jit
def direction_interactive( params, batch):


    key = jax.random.PRNGKey(42)
    model = custom_models['simple'](*(8,1))
    
    # model.init(key, jax.random.normal(key, (8,1000)))['params']
    # params = model.init(key, jax.random.normal(init_rng, (8,1000)))['params']


    loss_mix = 0.25
    X, Y, K = batch['X']['vector'], batch['Y'], batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)
    loss1 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))

    loss2 = jnp.empty((1,))
    
    for i in range(len(X)):
        # np.where instead of if
        if jnp.array(K['vector'][i]).size ==0:
            continue

        g = grad(predict_wrapper, argnums=2)(model, params, X[i])
        
        directional_derivative = jnp.dot(g, jnp.array(K['vector'][i]).T)
        
        sign = jnp.tanh(20.0 * directional_derivative)
        sign = sign * Y[i] * 2 - sign
        # print('shape: ',jnp.shape(jnp.abs(sign+1)))
        loss2 = jnp.vstack((loss2.T, jnp.abs(sign+1))).T #extend(jnp.abs(sign + 1))

    
    loss2 = jnp.mean(loss2)
    loss = (1 - loss_mix) * loss1 + loss_mix * loss2
    
    return loss, logits

def direction_interactive2( params, batch):


    key = jax.random.PRNGKey(42)
    model = custom_models['simple'](*(8,1))
    
    # model.init(key, jax.random.normal(key, (8,1000)))['params']
    # params = model.init(key, jax.random.normal(init_rng, (8,1000)))['params']


    loss_mix = 0.5


    X, Y, K = batch['X']['vector'], batch['Y'], batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)
    loss1 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))

    loss2 = jnp.empty((1,))

    Kvec = jnp.array(K['vector'])
    
    for i in range(len(X)):
        # np.where instead of if
        if jnp.any(jnp.isnan(jnp.array(K['vector'][i]))):
            continue

        g = grad(predict_wrapper, argnums=2)(model, params, X[i])
        
        directional_derivative = jnp.dot(g, K['vector'][i].T)
        
        
        sign = jnp.tanh(20.0 * directional_derivative)
        sign = sign * Y[i] * 2 - sign
        # print('shape: ',jnp.shape(jnp.abs(sign+1)))
        loss2 = jnp.vstack((loss2.T, jnp.abs(sign+1))).T #extend(jnp.abs(sign + 1))

    loss2 = jnp.mean(loss2)
    loss = (1 - loss_mix) * loss1 + loss_mix * jnp.mean(loss2)
    # print('Loss: ',loss)
    return loss, logits
    

def direction_interactive3(hyperparams, model, params, batch):

    X,Y,K = batch['X']['vector'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X).squeeze(axis=-1)  

    loss1 = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    loss2 = jnp.empty((0,1))

    for i,d in enumerate(X):
        for j,k in enumerate(K['vector'][i]):
            g = grad(predict_wrapper,argnums=2)(model, params, d)
            directional_derivative = jnp.dot(g,k.T) 

            sign = jnp.tanh(20.0*directional_derivative)
            sign = sign * Y[i] * 2 - sign
            

            loss2 = jnp.append(loss2,100*jnp.abs(sign+1))

            # print(f"LABEL: {Y[i]}")
            # print(f"PRECONV: {sign.primal}")
            # print(f"POSTCONV: {sign.primal}")
            # print(f"LOSS: {jnp.abs(sign+1).primal}")
        

    try:
        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*jnp.mean(loss2)
        
    except:

        loss = (1-hyperparams['loss_mix'])*loss1 + hyperparams['loss_mix']*0
    
    # print(f"LOSS: {loss.primal}")

    return loss, logits


def gradient_supervision_archive(hyperparams, model, params, batch):

    X,Y,K = batch['X'],batch['Y'],batch['K']

    logits = model.apply({'params': params}, X).squeeze(axis=-1)  
    
    loss_ce = np.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    loss_gs = jnp.empty((0,1))     

    # loss = 1 - (gi.gˆi) / (||gi|| ||gˆi||)

    for i,d in enumerate(X):
        
        g_x = K['vector'][i] * K['magnitude'][i]
        g_y = grad(predict_wrapper,argnums=2)(model, params, d)
        # make unit vectors, inspect
        # loss_i = 1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)+1e-5)
        # track cross product over time
        
        dot = jnp.dot(g_x,g_y)
        cross = jnp.cross(g_x,g_y)

        # print(f'Dot: {dot.primal}\nCross: {cross.primal}')
        # loss_i = jnp.where(jnp.isnan(loss_i),0,loss_i)
        # loss_gs = jnp.append(loss_gs,1-jnp.dot(g_x,g_y)/(jnp.cross(g_x,g_y)))
        try:
            loss_gs = jnp.append(loss_gs,1-dot/cross)
        except:
            print(f'Dot: {dot.primal}\nCross: {cross.primal}')
            print(f'G_X: {g_x}\nG_Y: {g_y.primal}')
    
    loss = (1-hyperparams['loss_mix'])*loss_ce + hyperparams['loss_mix']*jnp.mean(loss_gs)
    
    return loss, logits
    # return jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=label)), logits

loss_functions = {'direction':direction,
                  'loss_wrapper':loss_wrapper,
                  'direction_new':direction_new,
             'cross_entropy':cross_entropy,
             'cross_entropy_batch':cross_entropy_batch,
             'cross_entropy_l2':cross_entropy_l2,
             'gradient_supervision':gradient_supervision,
             'gradient_supervision_basic':gradient_supervision_basic,
             'direction_interactive': direction_interactive,
             'direction_interactive2': direction_interactive2,
             'direction_interactive3': direction_interactive3,
             'direction_interactive_vectorized': direction_interactive_vectorized}