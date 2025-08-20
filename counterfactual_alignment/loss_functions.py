

from counterfactual_alignment.utilities import (predict_wrapper,
                                                predict_wrapper2,
                                                predict_wrapper_v2, 
                                                get_unit_vec,
                                                get_max_dimension_and_index, 
                                                jagged_lists_to_array,
                                                convert_to_list_of_lists, 
                                                create_identical_matrix,
                                                classifier_apply,
                                                embed_and_average,
                                                predict_wrapper_embedding,
                                                cosine_distance_batch,
                                                embed_and_average_batchK)
from counterfactual_alignment.custom_models import custom_models
import numpy as np
import sys
import jax.numpy as jnp
import jax
from jax import grad, vmap
from flax import linen as nn

import torch
import optax
import yaml
import os
from sklearn.metrics.pairwise import cosine_similarity
jax.config.update("jax_debug_nans", True)


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



def gradient_supervision_basic_archive(params, model, batch, rng, alpha = 20):
    X,Y,K = batch['X'],batch['Y'],batch['K']

    print("X: ",np.shape(X),"| K: ",np.shape(K['vector']))
    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)
    
    # Now call the batched gradient function on the entire input array
    g_y = batched_grad_fn(params, model, X, rng)
    EPS = 1e-8
    alt_loss = []

    for i in range(len(X)):

        g_y_i = grad_fn(params, model, X[i], rng)
        g_hat = K['vector'][i][0]*-1

        cosine_sim_i = (jnp.dot(g_y_i, g_hat) /
                              (jnp.linalg.norm(g_y_i) * jnp.linalg.norm(g_hat) + EPS))
        loss_i = 1 - cosine_sim_i
        
        alt_loss.append(loss_i)
        
        if False:
            print("x:",X[i])
            print("y:",Y[i])
            jax.debug.print('g_y_i {}',g_y_i)
            jax.debug.print('g_hat {}',g_hat)        
            jax.debug.print('cosine similarity: {}',cosine_sim_i)
            jax.debug.print('loss_i {}',loss_i)

    alt_loss = jnp.mean(jnp.array(alt_loss))

    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))

    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(K['vector'])
    
    
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
    
    return loss



def gradient_supervision_basic(params, model, batch, rng, α = 20.0):
    X,Y,K = batch['X'],batch['Y'],batch['K']

    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    grad_fn = grad(predict_wrapper_v2,argnums=1, allow_int=False)
    
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, 0, None),out_axes=1)
    
    
    # Now call the batched gradient function on the entire input array
    
    # g_y = batched_grad_fn(params, X, rng) * (2*jnp.array(Y) - 1)[:,jnp.newaxis].T
    g_y = batched_grad_fn(params, X, rng) 
    
    EPS = 1e-8
    # alt_loss = []

    

    # for i in range(len(X)):


    #     g_y_i = grad_fn(params, X[i], rng)
    #     g_hat = K['vector'][i][0]*-1

    #     cosine_sim_i = (jnp.dot(g_y_i, g_hat) /
    #                           (jnp.linalg.norm(g_y_i) * jnp.linalg.norm(g_hat) + EPS))
    #     loss_i = 1 - cosine_sim_i
        
    #     alt_loss.append(loss_i)
        
    #     if True:
    #         print("x:",X[i])
    #         print("y:",Y[i])
    #         jax.debug.print('g_y_i {}',g_y_i)
    #         jax.debug.print('g_hat {}',g_hat)        
    #         jax.debug.print('cosine similarity: {}',cosine_sim_i)
    #         jax.debug.print('loss_i {}',loss_i)

    # alt_loss = jnp.mean(jnp.array(alt_loss))

    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    k_vector = jnp.multiply(jnp.array(K['vector']),jnp.array(K['magnitude']).reshape(-1,1,1))
    # X2 = X[:,jnp.newaxis] + kvec
    # k_coord = X2 - X[:,jnp.newaxis]                          

    # cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(-1*K['vector'])
    cosine_diff = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(k_vector)
        
    gs_loss = jnp.mean(jnp.array(cosine_diff))


    
    if False:
        jax.debug.print('X {}',X)
        # jax.debug.print('KVEC {}',k_vector)
        jax.debug.print('G_Y {}',g_y.T)
        # jax.debug.print('cos_diff {}',cosine_diff)
        # jax.debug.print('gs_loss {}',gs_loss)
        # jax.debug.print('ce_loss {}',ce_loss)

    # # loss = ce_loss + gs_loss
    # loss = alt_loss

    # jax.debug.print("Gradient Loss: {x}",x=gs_loss)
    
    return ce_loss + α * gs_loss


def gradient_supervision(params, model, batch, rng, α = 20):
    
    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=np.array(Y)))
    # Y = Y-1
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    grad_fn = grad(predict_wrapper,argnums=2,allow_int=True)
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)
    
    X_embedded = jnp.take(params['embed']['embedding'],np.array(X))
    print(np.shape(X_embedded))
    # Now call the batched gradient function on the entire input array
    g_y = batched_grad_fn(params, model, np.array(X), rng)

    EPS = 1e-8
    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))

   
    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(K['vector'])


    gs_loss = jnp.mean(jnp.array(cosine_sim))

    if False:
        jax.debug.print('g_y {}',g_y)
        jax.debug.print('cosim {}',cosine_sim)
        jax.debug.print('gs_loss {}',gs_loss)
        jax.debug.print('ce_loss {}',ce_loss)

    loss = ce_loss + α * gs_loss
    # jax.debug.print("Gradient Loss: {x}",x=gs_loss)
    
    return loss, logits


def gradient_supervision_embedding_archive(params, model, batch, rng, α=20):
    X, Y, K = batch['X'], batch['Y'], batch['K']

    # Forward pass with original model (int tokens input)
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})  
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=jnp.array(Y)))
    # print("Embeddings shape: ",np.shape(embeddings))
    params_linear = {'linear1': params['linear1']}
    # print("KVEC: ",K['vector'])
    # print("KVECSHAPE: ",np.shape(K['vector']))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    k_embs = k_embs.reshape(batch_size, k_count, -1)
    # _, k_embs = jax.vmap(model.apply,in_axes = (None,1))({'params': params}, K['vector'], train=True, rngs={'dropout': rng})
    # Get averaged embeddings for batch
    # x_embs = embed_and_average(params, X)  # shape: (batch_size, embed_dim)
    # k_embs = jax.vmap(embed_and_average,in_axes=(None,0))(params,K['vector'])  # shape: (batch_size, k_count, embed_dim)
    # y_grads = jax.vmap(jax.grad(predict_wrapper_embedding,argnums=1),in_axes=(None,0))(params,x_embs)
    
    # Attempt 2
    # single_grad_sigmoid = jax.grad(lambda p, x: jax.nn.sigmoid(classifier_apply(p, x)), argnums=1)
    # y_grads = jax.vmap(single_grad_sigmoid, in_axes=(None, 0))(params, x_embs)
    
    # Attempt 3
    # vmap over the batch to get per-sample grads
    y_grads = jax.vmap(
        jax.grad(predict_wrapper_embedding, argnums=1),
        in_axes=(None, 0)
    )(params_linear, embeddings)

    # y_grads = jax.vmap(jax.grad(lambda p, x: jnp.mean(predict_wrapper_embedding(p, x)), argnums=1),in_axes=(None,0))(params,x_embs)
    k_vecs = k_embs - embeddings[ :,jnp.newaxis, :]  # shape: (batch_size, k_count, embed_dim)
    
    cosine_distance = cosine_distance_batch(y_grads,k_vecs)
    gs_loss = jnp.mean(cosine_distance)
    


    jax.debug.print("X shape: {x}",x=np.shape(embeddings))
    jax.debug.print("X: {x}",x=X[0] - X[1])
    jax.debug.print("X: {x}",x=embeddings[0] - embeddings[1])

    jax.debug.print("K shape: {x}",x=np.shape(k_embs))
    jax.debug.print("K: {x}",x=K['vector'][0][0] - K['vector'][1][0])
    jax.debug.print("K: {x}",x=k_embs[0][0] - k_embs[1][0])

    jax.debug.print("Y Grads: {x}",x=np.shape(y_grads))
    jax.debug.print("Y Grads: {x}",x=y_grads[0] - y_grads[1])

    if jnp.any(jnp.isnan(y_grads)):
        jax.debug.print("Y Grads: {x}",x=y_grads)
        jax.debug.print("X vectors: {x}",x=embeddings)
        jax.debug.print("K vectors: {x}",x=k_vecs)
        jax.debug.print("cosine distance: {x}",x=cosine_distance)
        jax.debug.print("cosine distance mean: {x}",x=gs_loss)
        raise ValueError()

    # # avg_embed_K = embed_and_average_batchK(params, K['vector'])  # shape: (batch_size, k_count, embed_dim)
    # # Vectorized gradient of predict_on_avg_embed w.r.t avg_embed
    # grad_fn = jax.vmap(jax.grad(predict_wrapper_embedding,has_aux=True,argnums=1),in_axes=(None,0))
    # # grad_fn = jax.vmap(jax.grad(classifier_apply,has_aux=True,argnums=1))

    # g_y = grad_fn(params,x_embs)  # shape: (batch_size, embed_dim)
    
    
    
    

    

    # EPS = 1e-8


    # # def cosine_distance(a_row, b_col):
    # #     return 1 - (jnp.dot(a_row, b_col) / (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS))
    
    # # map_cosine = jax.vmap(lambda a_row, b_col: cosine_distance(a_row, b_col), in_axes=(0, 1))

    # # Compute cosine similarity between gradient g_y and K_avg
    # # cosine_sim = cosine_similarity_batch(g_y, avg_embed_K)  # (batch, num_K)
    
    # g_y_norm = g_y / jnp.linalg.norm(g_y + EPS,axis=-1, keepdims=True)
    # k_vec_norm = k_vec / jnp.linalg.norm(k_vec + EPS,axis=-1, keepdims=True)
    
    # cosine_sim = jnp.mean(g_y_norm * k_vec_norm, axis=-1)  # shape: (batch_size, k_count)
    # # K['vector']: shape (batch_size, k_count, embed_dim)
    # # Compute cosine similarity between g_y and each counterfactual vector in K

    # # cosine_sim = jax.vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(K['vector'])
    
    # # cosine_sim = jax.vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(avg_embed_K)
    
    # gs_loss = jnp.mean(cosine_sim,axis=-1)
    

    # gs_loss = gs_loss.squeeze(axis=-1)  # shape: (batch_size,)

    
    
    # print("GH_LOSS: ",gs_loss)
    
    loss = ce_loss + α * gs_loss
    
    return gs_loss



def gradient_supervision_embedding_archive2(params, model, batch, rng, α=20):
    
    ce_loss = cross_entropy(params, model, batch, rng)
    X, Y, K = batch['X'], batch['Y'], batch['K']
    
    # Forward pass with original model (int tokens input)
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})  
    
    # if jnp.any(jnp.isnan(logits)):
    #     # jax.debug.breakpoint()
    #     jax.debug.print(params)
    #     raise ValueError()
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=jnp.array(Y)))
    # print("Embeddings shape: ",np.shape(embeddings))
    params_linear = {'linear1': params['linear1']}
    # print("KVEC: ",K['vector'])
    # print("KVECSHAPE: ",np.shape(K['vector']))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    k_embs = k_embs.reshape(batch_size, k_count, -1)
    
    # k_vecs = k_embs - embeddings[ :,jnp.newaxis, :] + 1e-8  # shape: (batch_size, k_count, embed_dim)
    k_vecs=k_embs
    grad_fn = grad(predict_wrapper_embedding, argnums=1)
    batched_grad_fn = vmap(grad_fn, in_axes=(None, 0), out_axes=1)
    gslosslist = []
    for i,x in enumerate(embeddings):
        g_y_i = grad(predict_wrapper_embedding,argnums=1)(params_linear,x)

        for k in k_vecs[i]:
            jax.debug.print("X: {x},\nXNORM: {y}",x=x,y=jnp.linalg.norm(x))
            jax.debug.print("K: {x},\nKNORM: {y}",x=k,y=jnp.linalg.norm(k))
            cosdist = 1 - (jnp.dot(x, k) /
                              (jnp.linalg.norm(x) * jnp.linalg.norm(k)+1e-8))
            gslosslist.append(cosdist)
        jax.debug.print("GYI: {x}",x=g_y_i)
            

    # print('SHape of embs: ',np.shape(embeddings))
    # print("X: ",embeddings[0],
    #     "\nKe: ",k_embs[0][0],
    #     "\nKe_HAT: ",k_embs[0][0]*-1,
    #     "\nKv: ",k_vecs[0][0],
    #     "\nKv_HAT: ",k_vecs[0][0]*-1)
    # jax.debug.print(
    # "X: {}\nKe: {}\nKe_HAT: {}\nKv: {}\nKv_HAT: {}",
    # embeddings[0],
    # k_embs[0][0],
    # k_embs[0][0] * -1,
    # k_vecs[0][0],
    # k_vecs[0][0] * -1
    # )

    g_y = batched_grad_fn(params_linear,embeddings)
    # g_y = batched_grad_fn({'params': params}, X, train=True, rngs={'dropout': rng})
    
    EPS = 1e-8
    
    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
    

    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(k_vecs) + 1e-8
    jax.debug.print("KVECS: {x}",x=k_vecs)
    jax.debug.print("GY: {x}",x=g_y)
    jax.debug.print("COSIM: {x}",x=cosine_sim)
    gs_loss = jnp.mean(jnp.array(cosine_sim))
    jax.debug.print("GSLOSS: {x}",x=jnp.mean(jnp.array(cosine_sim)))
    # y_grads = jax.vmap(jax.grad(lambda p, x: jnp.mean(predict_wrapper_embedding(p, x)), argnums=1),in_axes=(None,0))(params,x_embs)
    
    
    # cosine_distance = cosine_distance_batch(y_grads,k_vecs)
    # gs_loss = jnp.mean(cosine_distance)
        

    # if jnp.any(jnp.isnan(y_grads)):
    #     jax.debug.print("X shape: {x}",x=np.shape(embeddings))
    #     # jax.debug.print("X: {x}",x=X[0] - X[1])
    #     jax.debug.print("X_0: {x}",x=embeddings[0])
    #     jax.debug.print("X_diff: {x}",x=np.mean(embeddings[0] - embeddings[1]))

    #     jax.debug.print("K shape: {x}",x=np.shape(k_embs))
    #     jax.debug.print("K_0: {x}",x=k_embs[0])
    #     # jax.debug.print("K: {x}",x=K['vector'][0][0] - K['vector'][1][0])
    #     jax.debug.print("K_diff: {x}",x=np.mean(k_embs[0][0] - k_embs[1][0]))

    #     jax.debug.print("Y Grads: {x}",x=np.shape(y_grads))
    #     jax.debug.print("Y Grads: {x}",x=y_grads[0])
    #     jax.debug.print("Y Grads diff: {x}",x=np.mean(y_grads[0] - y_grads[1]))
        
    #     raise ValueError("NaNs on the loose")
    
        
    #     # jax.debug.print("Y Grads: {x}",x=y_grads)
    #     # jax.debug.print("X vectors: {x}",x=embeddings)
    #     # jax.debug.print("K vectors: {x}",x=k_vecs)

    # if False:
    #     # jax.debug.print("cosine distance: {x}",x=cosine_distance)
    #     jax.debug.print("cosine distance mean: {x}",x=gs_loss)
    #     # raise ValueError()

    # loss = 
    jax.debug.print('cs loss :{x}',x=ce_loss)
    jax.debug.print("GSLIST : {x}",x =jnp.mean(jnp.array(gslosslist)))
    # return ce_loss + α * gs_loss

    return ce_loss + jnp.mean(jnp.array(gslosslist))



def gradient_supervision_embedding(params, model, batch, rng, α=20):
    
    X, Y, K = batch['X'], batch['Y'], batch['K']
    """
    ce_loss = cross_entropy(params, model, batch, rng)
    
    # Forward pass with original model (int tokens input)

    _, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    # jax.debug.print("PARAMS: {x}",x=params)
    # grads = []
    """
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    # jax.debug.print("PREDS : {x} | LABELS: {y} ",x=nn.sigmoid(logits),y = jnp.array(Y))
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=jnp.array(Y)))
    # jax.debug.print("PARAMS: {x},{y}",x = params['linear1'],y=np.shape(params['linear1']['kernel']))
    # for x in embeddings:
    #     jax.debug.print("Output: {x}",x=predict_wrapper_embedding({'linear1': params['linear1']},x))
    #     g_y_x = grad(predict_wrapper_embedding,argnums=1)({'linear1': params['linear1']},x)
        
    #     jax.debug.print("GYX: {x}",x=g_y_x)
    #     grads.append(g_y_x)


    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    k_embs = k_embs.reshape(batch_size, k_count, -1)

    
    k_vecs = k_embs - embeddings[ :,jnp.newaxis, :] + 1e-8 # shape: (batch_size, k_count, embed_dim)
    # k_vecs=k_embs
    
    params_linear = {'linear1': params['linear1']}
    # k_vecs=k_embs
    grad_fn = grad(predict_wrapper_embedding, argnums=1)
    # batched_grad_fn = vmap(grad_fn, in_axes=(None, 0), out_axes=1)
    batched_grad_fn = vmap(grad_fn, in_axes=(None, 0))
    
    g_y = batched_grad_fn(params_linear,embeddings) * (2*jnp.array(Y) - 1)[:,jnp.newaxis]
    # g_y = batched_grad_fn({'params': params}, X, train=True, rngs={'dropout': rng})
    # g_y = jnp.array(grads)

    # jax.debug.print("GY!: {x} | GY2 {y}",x=jnp.sum(g_y1),y=jnp.sum(g_y))
    EPS = 1e-8
    """
    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
    # cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y.T), in_axes=1)(k_vecs*-1) + 1e-8
    cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y.T), in_axes=1)(k_vecs) + 1e-8
    gs_loss = jnp.mean(jnp.array(cosine_sim))
    """
    def cosine_distance(a, b, eps=1e-8):
        return 1 - (jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + eps))

    # Compare each k_{i,j} to g_y[i]
    per_example_loss = vmap(
        lambda k_i, g_i: jnp.mean(vmap(lambda kv: cosine_distance(kv, g_i))(k_i)),
        in_axes=(0, 0)
    )(-k_vecs, g_y)

    gs_loss = jnp.mean(per_example_loss)

    # jax.debug.print("COSIM: {x}",x=cosine_sim)
    
    # jax.debug.print("GSLOSS: {x}",x=gs_loss)
    # # jax.debug.print("GSLIST : {x}",x =jnp.mean(jnp.array(gslosslist)))
    # jax.debug.print('cs loss :{x}',x=ce_loss)
    
    return ce_loss + α * gs_loss
    # return ce_loss

    # return ce_loss + α*jnp.mean(jnp.array(gslosslist))
# ---------------------


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


def cross_entropy(params, model, batch, rng, config=None):
    X,Y = np.array(batch['X']),np.array(batch['Y'])
    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    preds = nn.sigmoid(logits)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=np.array(Y)))
    # jax.debug.print('cs loss :{x}',x=loss)
    # jax.debug.print('cs logits :{x}',x=logits)
    # jax.debug.print('cs preds :{x}',x=preds)
    # jax.debug.print('labels :{x}',x=np.array(Y))
    return loss


def predict(params, model, x, rng, train = True):
    return jax.nn.sigmoid(model.apply({'params': params}, x, train=train, rngs={'dropout': rng})) #.squeeze(axis=-1)  
    # return y.squeeze(axis=-1)  



def directional_loss(theta, y, scale=10.0):
    # Want theta > 0 when y == 0; theta < 0 when y == 1
    # desired sign: +1 for y=0, -1 for y=1
    desired_sign = 1.0 - 2.0 * y
    violation = theta * desired_sign

    # If violation > 0 → correct direction → loss = 0
    # If violation < 0 → wrong direction → apply penalty
    return jax.nn.softplus(-scale * violation) / scale
    # return violation


def predict(params, model, x, rng, train=True):
    logits = model.apply({'params': params}, x, train=train, rngs={'dropout': rng})
    return jax.nn.sigmoid(logits).squeeze()

def grad_predict(params, model, x, rng):
    return grad(predict, argnums=2)(params, model, x, rng, train=False)

batched_grad_predict = vmap(grad_predict, in_axes=(None, None, 0, 0))

def directional_loss(theta, y, scale=10.0):
    desired_sign = 1.0 - 2.0 * y  # +1 for y=0, -1 for y=1
    violation = theta * desired_sign
    return jax.nn.softplus(-scale * violation) / scale

def direction(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']  # shapes: (N,D), (N,), (N,n,D)
    rngs = jax.random.split(rng, X.shape[0])

    g = batched_grad_predict(params, model, X, rngs)  # (N,D)

    K_exp = jnp.expand_dims(K, 1)  # (N,1,D)
    directional_derivative = jnp.einsum('ijk,ik->ij', K_exp, g)

    theta = directional_derivative[:, 0]  # first counterfactual direction
    loss = directional_loss(theta, Y)
    return jnp.mean(loss)


from jax import jvp

def direction23(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']
    N = X.shape[0]
    rngs = jax.random.split(rng, N)

    def model_output(x, rng_i):
        return predict(params, model, x, rng_i).squeeze()

    def directional_grad(x, k, rng_i):
        val, directional_derivative = jvp(lambda x_: model_output(x_, rng_i), (x,), (k,))
        return directional_derivative

    dd = vmap(directional_grad)(X, K, rngs)  # shape (N,)
    
    desired_sign = 1.0 - 2.0 * Y
    violation = dd * desired_sign
    loss = jax.nn.relu(-violation)

    return jnp.mean(loss)




def direction2(params, model, batch, rng, config=None):

    X,Y,K = batch['X'],batch['Y'],batch['K']
    rngs = jax.random.split(rng, X.shape[0])

    grad_fn = grad(predict,argnums=2)
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, 0),out_axes=0)

    # Now call the batched gradient function on the entire input array
    
    g = batched_grad_fn(params, model, X, rngs)
    
    
    # The following could be made less complicated if we assume only counterfactual per example - as it is, it allows for K to be 2D
    # map_dd = vmap(lambda a_row, b_col: jnp.dot(a_row, b_col), in_axes=(0, 1))
    # expand K dims to fit with this assumption
    
    K = jnp.expand_dims(K, 1)  # shape (N, 1, D)    
    directional_derivative = jnp.einsum('ijk,ik->ij', K, g)

    sign = jnp.sign(directional_derivative)
    loss = directional_loss(sign[:,0],Y)
  
    # print('\n\n\n\nK',K)
    # jax.debug.print("g: {x}",x=g)
    # jax.debug.print("gshape: {x}",x=np.shape(g))
    # jax.debug.print("dd shape: {x}",x=np.shape(directional_derivative))
    # jax.debug.print("dd: {x}",x=directional_derivative)
    jax.debug.print("sign: {x}",x=sign)
    print('Y',Y)
    jax.debug.print("loss: {x}",x=loss)
    

    return jnp.mean(loss)



def direction_wrapper(params, model, batch, rng, config=None):
    return direction(params,model,batch['additional'],rng)

def cross_entropy_wrapper(params, model, batch, rng, config=None):
    return cross_entropy(params,model,batch['original'],rng)

def combined_loss(params, model, batch, rng, α=0.5):

    ce_loss = cross_entropy_wrapper(params,model,batch,rng)

    d_loss = direction_wrapper(params,model,batch,rng)
    
    loss = (1 - α) * ce_loss + α * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(α)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss



def combined_loss_imdb(params, model, batch, rng, config):

    α = config['hyperparams']['loss_mix']

    for key,_ in batch.items():
        print(key,batch[key].items())

    ce_loss = cross_entropy(params,model,batch['original'],rng)

    d_loss = direction(params,model,batch['additional'],rng)
    
    loss = (1 - α) * ce_loss + α * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(α)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss


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
                  'combined_loss':combined_loss,
                  'combined_loss_imdb':combined_loss_imdb,
                  'direction_new':direction_new,
             'cross_entropy':cross_entropy,
             'cross_entropy_batch':cross_entropy_batch,
             'cross_entropy_l2':cross_entropy_l2,
             'gradient_supervision':gradient_supervision,
             'gradient_supervision_embedding':gradient_supervision_embedding,
             'gradient_supervision_basic':gradient_supervision_basic,
             'direction_interactive': direction_interactive,
             'direction_interactive2': direction_interactive2,
             'direction_interactive3': direction_interactive3,
             'direction_interactive_vectorized': direction_interactive_vectorized}