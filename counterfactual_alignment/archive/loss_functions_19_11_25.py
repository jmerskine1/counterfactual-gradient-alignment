

from counterfactual_alignment.utilities import (predict_wrapper,
                                                predict_wrapper2,
                                                predict_wrapper_v2, 
                                                get_unit_vec,
                                                jnp_get_unit_vec,
                                                get_max_dimension_and_index, 
                                                jagged_lists_to_array,
                                                convert_to_list_of_lists, 
                                                pad_and_mask,
                                                create_identical_matrix,
                                                classifier_apply,
                                                embed_and_average,
                                                predict_wrapper_embedding,
                                                embedding_only,
                                                cosine_distance_batch,
                                                embed_and_average_batchK)
from counterfactual_alignment.custom_models import custom_models, MulticlassEmbeddingOnlyModel
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


from flax import traverse_util

def l2_regularization(params):
    # flatten param tree into dict
    flat_params = traverse_util.flatten_dict(params)
    # sum over all parameters
    return sum([jnp.sum(jnp.square(p)) for p in flat_params.values()])


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

    alpha = config['hyperparams']['loss_mix'] 
    
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
                
                
    
    loss = loss + alpha*jnp.mean(jnp.array(loss_gs))
    
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
    loss = (1-alpha)*ce_loss + alpha*d_loss

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



def gradient_supervision_basic(params, model, batch, rng, alpha = 20):
    X,Y,K = batch['X'],batch['Y'],batch['K']

    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    # batched_grad_fn = vmap(grad_fn, in_axes=(None, 0, None),out_axes=1)
    batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, None,0, None),out_axes=1)
    
    
    # Now call the batched gradient function on the entire input array
    
    # g_y = batched_grad_fn(params, X, rng) * -(2*jnp.array(Y) - 1)[:,jnp.newaxis].T
    # g_y = -batched_grad_fn(params, X, rng) 
    # g_y = -batched_grad_fn(params, X, rng) * -(2*jnp.array(Y) - 1)[:,jnp.newaxis].T
    # g_y = batched_grad_fn(params, X, rng) 
    g_y = batched_grad_fn(params, model, X, rng) * -(2*jnp.array(Y) - 1)[:,jnp.newaxis].T

    EPS = 1e-8


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
    
    return ce_loss + alpha * gs_loss
    # return gs_loss


def gradient_supervision(params, model, batch, rng, alpha = 20):
    
    X,Y,K = batch['X'],batch['Y'],batch['K']
    
    logits = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=np.array(Y)))
    # Y = Y-1
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    grad_fn = grad(predict_wrapper,argnums=2,allow_int=True)
    batched_grad_fn = vmap(grad_fn, in_axes=(None, None, 0, None),out_axes=1)
    
    X_embedded = jnp.take(params['embed']['embedding'],np.array(X))
    
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

    loss = ce_loss + alpha * gs_loss
    # jax.debug.print("Gradient Loss: {x}",x=gs_loss)
    
    return loss, logits


def gradient_supervision_embedding_archive(params, model, batch, rng, alpha=20):
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
    
    loss = ce_loss + alpha * gs_loss
    
    return gs_loss



def gradient_supervision_embedding_archive2(params, model, batch, rng, alpha=20):
    
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
    # return ce_loss + alpha * gs_loss

    return ce_loss + jnp.mean(jnp.array(gslosslist))



def gradient_supervision_embedding(params, model, batch, rng, alpha=1):
    
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
    # k_vecs = embeddings[ :,jnp.newaxis, :] - k_embs + 1e-8 # shape: (batch_size, k_count, embed_dim)
    # jax.debug.print("KSHAPE {}",np.shape(k_vecs))
    
    params_linear = {'linear1': params['linear1']}

    # k_vecs=k_embs
    grad_fn = grad(predict_wrapper_embedding, argnums=1,has_aux=True)
    # batched_grad_fn = vmap(grad_fn, in_axes=(None, 0), out_axes=1)
    batched_grad_fn = vmap(grad_fn, in_axes=(None, 0))
    
    # g_y = batched_grad_fn(params_linear,embeddings).T * -(2*jnp.array(Y) - 1)[:,jnp.newaxis].T
    rngs = jax.random.split(rng, embeddings.shape[0])
    g_y = batched_grad_predict(params_linear,embedding_only,embeddings,rngs) * (1 - 2*jnp.array(Y))[:,jnp.newaxis]
    # g_y = batched_grad_fn({'params': params}, X, train=True, rngs={'dropout': rng})
    # g_y = jnp.array(grads)

    # jax.debug.print("GY!: {x} | GY2 {y}",x=jnp.sum(g_y1),y=jnp.sum(g_y))
    EPS = 1e-8
    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
    # X2 = X[:,jnp.newaxis] + kvec
    # k_coord = X2 - X[:,jnp.newaxis]                          

    # cosine_sim = vmap(lambda K_slice: map_cosine(K_slice, g_y), in_axes=1)(-1*K['vector'])
    cosine_diff = vmap(lambda K_slice: map_cosine(K_slice, g_y.T), in_axes=1)(k_vecs)
        
    gs_loss = jnp.mean(jnp.array(cosine_diff))
    
    return (1 - alpha) * ce_loss + alpha * gs_loss
    
 


def multiclass_gradient_supervision(params, model, batch, rng, config=None,alpha=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)

    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    
    g_y = jac_map(params, model, X, rng)
    g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])

    EPS = 1e-8

    map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
                              (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
    cosine_diff = vmap(lambda K_slice: map_cosine(K_slice+EPS, g_y_2.T+EPS), in_axes=1)(K)

    # map_cosine = vmap(
    #     lambda a_row, b_col: 1 - jnp.dot(
    #         a_row / jnp.maximum(jnp.linalg.norm(a_row), EPS),
    #         b_col / jnp.maximum(jnp.linalg.norm(b_col), EPS)
    #     ),
    #     in_axes=(0, 1)
    # )   

    # cosine_diff = vmap(lambda K_slice: map_cosine(K_slice, g_y_2.T), in_axes=1)(K)

    return jnp.mean(jnp.array(cosine_diff))

# ---------------------
def multiclass_gradient_supervision_embedding_4_10_25(params, model, batch, rng, alpha=1):
    
    X, Y, K = batch['X'], batch['Y'], batch['K']
    # print("XTYPE: ",type(X), " | XSHAPE: ",np.shape(X))
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    
    k_embs = k_embs[:,jnp.newaxis,:]

    embedding_length = embeddings.shape[-1]
    embeddings_expanded = jnp.expand_dims(embeddings, axis=1)        # (batch, 1, vector_length)
    embeddings_expanded = jnp.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length
    embeddings_expanded = embeddings_expanded.reshape(batch_size * k_count, embedding_length)
    y_expanded = jnp.expand_dims(np.array(Y), axis=1)
    y_expanded = jnp.repeat(y_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    y_expanded = y_expanded.reshape(batch_size * k_count)
    

    # k_vecs = k_embs - embeddings[ :,jnp.newaxis, :] + 1e-8 # shape: (batch_size, k_count, embed_dim)
    
    params_linear = {'linear1': params['linear1']}
    # embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=np.shape(logits)[1])

    gs_loss = multiclass_gradient_supervision(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_embs},
                        },rng)
    
    return (1-alpha)*ce_loss + alpha * gs_loss


def multiclass_gradient_supervision_embedding(params, model, batch, rng, alpha=1):
    
    X, Y, K = np.array(batch['X']), batch['Y'], batch['K']
    # print("XTYPE: ",type(X), " | XSHAPE: ",np.shape(X))
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    
    k_embs = k_embs[:,jnp.newaxis,:]

    embedding_length = embeddings.shape[-1]
    embeddings_expanded = jnp.expand_dims(embeddings, axis=1)        # (batch, 1, vector_length)
    embeddings_expanded = jnp.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length
    embeddings_expanded = embeddings_expanded.reshape(batch_size * k_count, embedding_length)
    y_expanded = jnp.expand_dims(np.array(Y), axis=1)
    y_expanded = jnp.repeat(y_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    y_expanded = y_expanded.reshape(batch_size * k_count)
    

    # k_vecs = k_embs - embeddings[ :,jnp.newaxis, :] + 1e-8 # shape: (batch_size, k_count, embed_dim)
    
    params_linear = {'linear1': params['linear1']}
    # embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=np.shape(logits)[1])

    gs_loss = multiclass_gradient_supervision(params_linear,embedding_only,
                       {"X":embeddings_expanded,
                        'Y':y_expanded,
                        "K":{'vector':k_embs},
                        },rng)
    
    return (1-alpha)*ce_loss + alpha * gs_loss


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


def multiclass_cross_entropy(params, model, batch, rng, config=None, alpha = None):
    X,Y = np.array(batch['X']),np.array(batch['Y'])
    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    preds = nn.sigmoid(logits)
    loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    
    # jax.debug.print('cs logits :{x}',x=logits)
    # jax.debug.print('cs preds :{x}',x=preds)
    # jax.debug.print('labels :{x}',x=np.array(Y))

    return loss


# def multiclass_cross_entropy(params, model, batch, rng, config=None, alpha = None):
#     X,Y,K = np.array(batch['X']),np.array(batch['Y']),np.array(batch['K']['vector'])
    
#     logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
#     preds = nn.sigmoid(logits)
#     loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
#     # jax.debug.print('cs loss :{x}',x=loss)
#     # jax.debug.print('cs logits :{x}',x=logits)
#     # jax.debug.print('cs preds :{x}',x=preds)
#     # jax.debug.print('labels :{x}',x=np.array(Y))
    

#     jac_fn = jax.jacobian(predict_wrapper, argnums=2)
#     jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    
#     g_y = jac_map(params, model, X, rng)
#     g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])

#     batch_size, k_count = K.shape[:2]
#     # k_vectors_flat = K.reshape(batch_size * k_count, *K.shape[2:])
    
    
    
#     # k_embs = k_vectors_flat.reshape(batch_size, k_count, -1)
#     # k_vecs = X[ :,jnp.newaxis, :]- K + 1e-8 # shape: (batch_size, k_count, embed_dim)
#     k_vecs = K - X[ :,jnp.newaxis, :] + 1e-8 # shape: (batch_size, k_count, embed_dim)
#     print("X: ",X)
#     print("K: ",K)
#     jax.debug.print("g_y: {}",g_y)
#     jax.debug.print("gy2: {}",g_y_2)
#     jax.debug.print("k_vecs: {}",k_vecs)
#     EPS = 1e-8
#     map_cosine = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
#                               (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
#     cosine_diff = vmap(lambda K_slice: map_cosine(K_slice, g_y_2.T), in_axes=1)(k_vecs)

#     jax.debug.print('cosine diff: {}',cosine_diff)
#     # jax.debug.print("GRADEINT LOSS: {}",jnp.mean(jnp.array(cosine_diff)))
#     gs_loss =jnp.mean(jnp.array(cosine_diff))
#     jax.debug.print('gs loss: {}',gs_loss)
#     return loss


def cross_entropy(params, model, batch, rng, config=None,alpha=None):
    X,Y = np.array(batch['X']),np.array(batch['Y'])
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    labels = jnp.array(Y).reshape(-1, 1)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))
    # jax.debug.print("LOSS: {}",loss)
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
    logits,_ = model.apply({'params': params}, x, train=train, rngs={'dropout': rng})
    return jax.nn.sigmoid(logits).squeeze()

def grad_predict(params, model, x, rng):
    return grad(predict, argnums=2)(params, model, x, rng, train=False)

batched_grad_predict = vmap(grad_predict, in_axes=(None, None, 0, 0))

def directional_loss(theta, y, scale=10.0):
    desired_sign = 1.0 - 2.0 * y  # +1 for y=0, -1 for y=1
    violation = theta * desired_sign
    return jax.nn.softplus(-scale * violation) / scale


def direction_archive_2(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)
    
    rngs = jax.random.split(rng, X.shape[0])

    g = batched_grad_predict(params, model, X, rngs)  # (N,D)
    
    directional_derivative = jnp.einsum('ijk,ik->ij', K, g)

    dd = directional_derivative[:,0]
    sign = jnp.tanh(dd*10)
    # loss = sign*Y*2 - sign + 1 * abs(sign)

    desired_sign = 1.0 - 2.0 * jnp.array(Y).astype(jnp.float32)
    # desired_sign = -(sign*Y*2 - sign)
    violation = dd * desired_sign
    loss =  jax.nn.softplus(-10 * violation) / 10
    # loss2 = sign*Y*2 - sign + 1 * abs(sign)
    # loss2 = jnp.abs(sign - desired_sign)
    # jax.debug.print("X : {}",X)
    # jax.debug.print("K : {}",K)
    # jax.debug.print("DD : {}",dd)
    # jax.debug.print("SIGN : {}",sign)
    # jax.debug.print("DESIRED SIGN : {}",desired_sign)
    # jax.debug.print('Y: {}',Y)
    # jax.debug.print("DIRECTION LOSS: {}",loss)
    # jax.debug.print("DIRECTION LOSS2: {}",loss2)
    
    
    # sign = jnp.sign(directional_derivative)
    # loss = sign*Y[:,None]*2 - sign + 1 * abs(sign)
    # theta = directional_derivative[:, 0]  # first counterfactual direction
    # loss = directional_loss(theta, Y)

    # jax.debug.print("GRADIENT : {}",directional_derivative)
    # jax.debug.print("SIGN : {}",sign)
    # jax.debug.print('theta: {}',theta)
    # jax.debug.print('Y: {}',Y)
    # jax.debug.print("DIRECTION LOSS: {}",jnp.mean(loss))
    # jax.debug.print("DIRECTION NEW LOSS: {}",new_loss)
    return jnp.mean(loss)



def direction(params, model, batch, rng, config=None, alpha=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)
    
    rngs = jax.random.split(rng, X.shape[0])

    g = batched_grad_predict(params, model, X, rngs)  # (N,D)

    # directional_derivative = jnp.einsum('ijk,ik->ij', K, g)
    directional_derivative = jnp.sum(g * K, axis=1)
    
    # directional_derivative = g @ K.T
    
    desired_sign = -(1.0 - 2.0 * jnp.array(Y).astype(jnp.float32))
    # print(Y[0])

    loss = nn.softplus(10*directional_derivative*desired_sign)/10

    
    # dd = directional_derivative[:,0]
    # sign = jnp.tanh(dd*10)
    # # loss = sign*Y*2 - sign + 1 * abs(sign)

    
    # # desired_sign = -(sign*Y*2 - sign)
    # violation = dd * desired_sign
    # loss =  jax.nn.softplus(-10 * violation) / 10
    
    return jnp.mean(loss)



def multiclass_direction_thursday_02_08(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)
    
    rngs = jax.random.split(rng, X.shape[0])

    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    # grads = jac_map(state.params, model, points, key)  # (num_points, num_classes, 2)

    g_y = jac_map(params, model, X, rng)
    
    g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])
    # direction = K-X.reshape(np.shape(K)[0],np.shape(K)[1],np.shape(K)[2])
    direction= X[:,None,:]-K
    # direction_reshaped = direction.squeeze(axis=1)
    # print("X & K: ",np.shape(X),np.shape(K))
    # print("DIRECTION & GY: ",np.shape(direction),np.shape(g_y_2))
    # directional_derivative = g_y_2 @ 
    # directional_derivative_hardcoded = g_y_2 @ direction_reshaped.T
    directional_derivative = jnp.einsum('ik,ijk->ij', g_y_2, direction).squeeze(axis=1)
    # directional_derivative = jnp.einsum('ijk,ik->ij', K-X, g_y_2)

    sign = jnp.tanh(directional_derivative*10)
    # loss = sign*Y*2 - sign + 1 * abs(sign)
    
    # desired_sign = 1.0 - 2.0 * jnp.array(Y).astype(jnp.float32)
    
    # desired_sign = -(sign*Y*2 - sign)

    # desired_sign = -1
    # violation = directional_derivative * desired_sign
    # loss =  jax.nn.softplus(-10 * violation) / 10

    loss = (1 + sign)
    # loss = violation
    jax.debug.print("X: {}",X)
    jax.debug.print("G_Y: {}",g_y_2)
    jax.debug.print("K: {}",K)
    jax.debug.print("Directional derivative: {}",directional_derivative)
    jax.debug.print("SIGN: {}",sign)
    # jax.debug.print("DESIRED SIGN: {}",desired_sign)
    # jax.debug.print("violation: {}",violation)
    jax.debug.print("direction loss: {}",jnp.mean(loss))
    
    return jnp.mean(loss)



def multiclass_direction_4_10_25(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)

    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)

    g_y = jac_map(params, model, X, rng)
    g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])
    
    
    directional_derivative = jnp.sum(g_y_2 * K.squeeze(axis=1), axis=1)  # shape (N,)
    
    # sign = jnp.tanh(20.0*directional_derivative)

    # loss = 1 + sign
    loss = nn.softplus(directional_derivative)
    
    return jnp.mean(loss)

def multiclass_direction(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)

    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    
    g_y = jac_map(params, model, X, rng)
    g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])
    directional_derivative = jnp.sum(g_y_2 * K, axis=1)  # shape (N,)
    
    magnitude = jnp.linalg.norm(K - X,axis=1)
    sign = jnp.tanh(20.0*directional_derivative)

    # loss = 1 + sign
    # loss = jnp.max(sign*magnitude,0)
    # loss = jnp.max(sign,0)
    # loss = nn.relu(directional_derivative/magnitude)
    loss = nn.softplus(10*directional_derivative)/10
    
    # loss = nn.softplus(directional_derivative/magnitude)
    # jax.debug.print("ddshaoe: {}",directional_derivative.shape)
    # jax.debug.print("magntidue: {}",magnitude.shape)
    # jax.debug.print("loss: {}",loss)
    
    return jnp.mean(loss)

def direction_2(params, model, batch, rng, config=None):
    X, Y, K = batch['K']['origins'], batch['K']['labels'], batch['K']['vectors']   # shapes: (N,D), (N,), (N,n,D)
    K = K[:,jnp.newaxis,:]
    rngs = jax.random.split(rng, X.shape[0])

    g = batched_grad_predict(params, model, X, rngs)  # (N,D)
    
    directional_derivative = jnp.einsum('ijk,ik->ij', K, g)

    dd = directional_derivative[:,0]
    sign = jnp.tanh(dd*10)
    # loss = sign*Y*2 - sign + 1 * abs(sign)

    desired_sign = 1.0 - 2.0 * jnp.array(Y).astype(jnp.float32)
    # desired_sign = -(sign*Y*2 - sign)
    violation = dd * desired_sign
    loss =  jax.nn.softplus(-10 * violation) / 10
    
    return jnp.mean(loss)




from jax import jvp

def direction_oldcollate(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)
    
    rngs = jax.random.split(rng, X.shape[0])

    g = batched_grad_predict(params, model, X, rngs)  # (N,D)
    
    directional_derivative = jnp.einsum('ijk,ik->ij', K, g)

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

def combined_loss_archive(params, model, batch, rng, alpha=0.5):

    ce_loss = cross_entropy(params,model,batch,rng)

    d_loss = direction_2(params,model,batch,rng)
    
    loss = (1 - alpha) * ce_loss + alpha * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(alpha)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss


def combined_loss(params, model, batch, rng, alpha=0.5):

    ce_loss = cross_entropy(params,model,batch,rng)

    d_loss = direction(params,model,batch,rng)
    
    loss = (1 - alpha) * ce_loss + alpha * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(alpha)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss



def multiclass_combined_loss(params, model, batch, rng, alpha=0.5):

    ce_loss = multiclass_cross_entropy(params,model,batch,rng)

    d_loss = multiclass_direction(params,model,batch,rng)
    
    loss = (1 - alpha) * ce_loss + alpha * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(alpha)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss

def combined_loss_imdb(params, model, batch, rng, config):

    alpha = config['hyperparams']['loss_mix']

    for key,_ in batch.items():
        print(key,batch[key].items())

    ce_loss = cross_entropy(params,model,batch['original'],rng)
    params_linear = {'linear1': params['linear1']}
    d_loss = direction(params_linear,predict_wrapper_embedding,batch['additional'],rng)
    
    loss = (1 - alpha) * ce_loss + alpha * d_loss
    
    if False:
        jax.debug.print("D Loss: {x}",x=d_loss)
        jax.debug.print("CE Loss: {x}",x=ce_loss)
        print(alpha)
        jax.debug.print("Loss: {x}",x=loss)
    # return loss, logits
    return loss



def combined_loss_embedding(params, model, batch, rng, alpha=0.5):
    X, Y, K = batch['X'], batch['Y'], batch['K']
    
    ce_loss = cross_entropy(params, model, batch, rng, alpha=0.5)

    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})

    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    # k_embs = k_embs[:,jnp.newaxis,:]

    embedding_length = embeddings.shape[-1]
    embeddings_expanded = jnp.expand_dims(embeddings, axis=1)        # (batch, 1, vector_length)
    embeddings_expanded = jnp.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length
    embeddings_expanded = embeddings_expanded.reshape(batch_size * k_count, embedding_length)
    y_expanded = jnp.expand_dims(np.array(Y), axis=1)
    y_expanded = jnp.repeat(y_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    y_expanded = y_expanded.reshape(batch_size * k_count)
    
    k_direction = batch_unit_vector(embeddings_expanded,k_embs)

    params_linear = {'linear1': params['linear1']}

    d_loss = direction(params_linear,embedding_only,
                       {"X":embeddings_expanded,
                        'Y':y_expanded,
                        "K":{'vector':k_direction},
                        },rng)
    
    # d_loss = direction(params_linear,embedding_only,
    #                    {"X":embeddings,
    #                     'Y':Y,
    #                     "K":{'vector':k_embs},
    #                     },rng)
    
    
    return (1 - alpha) * ce_loss + alpha * d_loss


def allcombined_loss_embedding(params, model, batch, rng, alpha=0.5):
    
    X, Y, K = batch['X'], batch['Y'], batch['K']
    """
    ce_loss = cross_entropy(params, model, batch, rng)
    
    # Forward pass with original model (int tokens input)

    _, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    # jax.debug.print("PARAMS: {x}",x=params)
    # grads = []
    """
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=jnp.array(Y)))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    k_embs = k_embs[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    d_loss = direction(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_embs},
                        },rng)
    gs_loss = gradient_supervision(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_embs},
                        },rng)
    
    
    return (1-alpha)*ce_loss + alpha * (d_loss + gs_loss)



def batch_unit_vector(x1, x2, eps=1e-8):
    diff = x2 - x1+eps                        # shape (40,64)
    norms = jnp.linalg.norm(diff, axis=1, keepdims=True)  # shape (40,1)
    unit = diff / (norms + eps)           # shape (40,64)
    return unit


def multiclass_combined_loss_embedding_4_10_25(params, model, batch, rng, alpha=1):
    
    X, Y, K = batch['X'], batch['Y'], batch['K']
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    embeddings_expanded = np.expand_dims(embedding, axis=1)        # (batch, 1, vector_length)
    embeddings_expanded = np.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    """
    k_embs = k_embs[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_embs},
                        },rng)
    """
    
    k_direction = batch_unit_vector(embeddings,k_embs)
    k_direction = k_direction[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_direction},
                        },rng)
    
    return (1 - alpha) * ce_loss + alpha * d_loss

def multiclass_combined_loss_embedding(params, model, batch, rng, alpha=1):
    
    X, Y, K = np.array(batch['X']), batch['Y'], batch['K']
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    vector_length = jnp.array(X).shape[-1]
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    # jax.debug.print("K_EMBS: {}",k_embs.shape)
    embedding_length = embeddings.shape[-1]
    embeddings_expanded = jnp.expand_dims(embeddings, axis=1)        # (batch, 1, vector_length)
    embeddings_expanded = jnp.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length
    embeddings_expanded = embeddings_expanded.reshape(batch_size * k_count, embedding_length)
    
    y_expanded = jnp.expand_dims(np.array(Y), axis=1)
    y_expanded = jnp.repeat(y_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    y_expanded = y_expanded.reshape(batch_size * k_count)
    
    k_direction = batch_unit_vector(embeddings_expanded,k_embs)
    # k_direction = k_direction[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])

    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":embeddings_expanded,
                        'Y':y_expanded,
                        "K":{'vector':k_direction},
                        },rng)
    
    return (1 - alpha) * ce_loss + alpha * d_loss


def multiclass_allcombined_loss_embedding(params, model, batch, rng, alpha=1):
    
    X, Y, K = batch['X'], batch['Y'], batch['K']
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    
    batch_size, k_count = K['vector'].shape[:2]
    k_vectors_flat = K['vector'].reshape(batch_size * k_count, *K['vector'].shape[2:])
    _, k_embs = model.apply({'params': params}, k_vectors_flat, train=False, rngs={'dropout': rng})
    
    """
    k_embs = k_embs[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_embs},
                        },rng)
    """
    
    k_direction = batch_unit_vector(embeddings,k_embs)
    k_direction = k_direction[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_direction},
                        },rng)
    gs_loss = multiclass_gradient_supervision(params_linear,embedding_only,
                       {"X":embeddings,
                        'Y':Y,
                        "K":{'vector':k_direction},
                        },rng)
    
    return (1 - alpha) * ce_loss + alpha * (gs_loss+d_loss)

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


    # Compute directional derivative loss (only matters if alpha > 0)
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
    alpha = config['hyperparams']['loss_mix']
    ce_loss = (1 - alpha) * ce_loss
    jax.debug.print("CE_LOSS: {x}",x = ce_loss)
    print("ALPHA",alpha)    
    loss = ce_loss + alpha * d_loss

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
    

    alpha = config['hyperparams']['loss_mix']
    ce_loss = (1 - alpha) * ce_loss
    jax.debug.print("CE_LOSS: {x}",x = ce_loss)
    print("ALPHA",alpha)    
    loss = ce_loss + alpha * d_loss
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

    alpha = 0.5

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

    loss = (1-alpha)*loss + alpha*jnp.mean(d_loss)
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

loss_functions =   {'direction':direction,
                    'combined_loss':combined_loss,
                    'multiclass_combined_loss':multiclass_combined_loss,
                    'combined_loss_embedding':combined_loss_embedding,
                    'multiclass_combined_loss_embedding':multiclass_combined_loss_embedding,
                    'multiclass_allcombined_loss_embedding':multiclass_allcombined_loss_embedding,
                    'combined_loss_imdb':combined_loss_imdb,
                    'direction_new':direction_new,
                    'cross_entropy':cross_entropy,
                    'multiclass_cross_entropy':multiclass_cross_entropy,
                    'cross_entropy_batch':cross_entropy_batch,
                    'cross_entropy_l2':cross_entropy_l2,
                    'gradient_supervision':gradient_supervision,
                    'gradient_supervision_embedding':gradient_supervision_embedding,
                    'multiclass_gradient_supervision':multiclass_gradient_supervision,
                    'multiclass_gradient_supervision_embedding':multiclass_gradient_supervision_embedding,
                    'gradient_supervision_basic':gradient_supervision_basic,
                    'direction_interactive': direction_interactive,
                    'direction_interactive2': direction_interactive2,
                    'direction_interactive3': direction_interactive3,
                    'direction_interactive_vectorized': direction_interactive_vectorized}