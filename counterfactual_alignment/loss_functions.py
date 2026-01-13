

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

from counterfactual_alignment.direction_class import GeometricDirectionalLoss   
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

def gradient_supervision_basic(params, model, batch, rng, alpha = 20):
    X,Y,K = batch['X'],batch['Y'],batch['K']

    
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng}) #.squeeze(axis=-1)  
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=Y))
    
    grad_fn = grad(predict_wrapper,argnums=2, allow_int=False)
    
    # Vectorize the gradient function over the batch of inputs using jax.vmap
    # batched_grad_fn = vmap(grad_fn, in_axes=(None, 0, None),out_axes=1)
    batched_grad_fn = jax.vmap(grad_fn, in_axes=(None, None,0, None),out_axes=1)
    
    
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
    
    ce_loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=jnp.array(Y)))

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

    return jnp.mean(jnp.array(cosine_diff))



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



def multiclass_split_gs_embedding(params, model, batch, rng, alpha=1):
    
    X, Y, K = np.array(batch['X']), batch['Y'], batch['K']
    # print("XTYPE: ",type(X), " | XSHAPE: ",np.shape(X))
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    dataset = {}
    for key,val in K.items():
        dataset[key] = jnp.concatenate(val)


    _, k_embs = model.apply({'params': params}, dataset['K'], train=False, rngs={'dropout': rng})
    _, x_embs = model.apply({'params': params}, dataset['X'], train=False, rngs={'dropout': rng})
    # k_vecs = k_embs - embeddings[ :,jnp.newaxis, :] + 1e-8 # shape: (batch_size, k_count, embed_dim)
    
    params_linear = {'linear1': params['linear1']}
    # embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=np.shape(logits)[1])

    gs_loss = multiclass_gradient_supervision(params_linear,embedding_only,
                       {"X":x_embs,
                        'Y':dataset['Y'],
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




def cross_entropy(params, model, batch, rng, config=None,alpha=None):
    X,Y = np.array(batch['X']),np.array(batch['Y'])
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    labels = jnp.array(Y).reshape(-1, 1)
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels))
    # jax.debug.print("LOSS: {}",loss)
    # jax.debug.print('ce logits :{x}',x=logits)
    # jax.debug.print('ce labels :{x}',x=labels)
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


def get_uncertainty_score(logits, temperature=1.0):
    """
    Higher score = High uncertainty (similar logits)
    Lower score = High confidence (one logit dominates)
    """
    # 1. Apply Temperature scaling (optional)
    # Lower temperature makes the distribution sharper
    logits = logits / temperature
    
    # 2. Softmax transformation for numerical stability
    e_x = jnp.exp(logits - jnp.max(logits, axis=1, keepdims=True))
    probs = e_x / e_x.sum(axis=1, keepdims=True)
    
    # 3. Calculate Shannon Entropy: -sum(p * log(p))
    # We use a small epsilon to prevent log(0)
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-12), axis=1)
    
    return entropy

# Example usage with your data:
# uncertainty = get_uncertainty_score(your_logits_array)

def multiclass_direction(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']   # shapes: (N,D), (N,), (N,n,D)

    """"""
    # X,Y = np.array(batch['X']),np.array(batch['Y'])
    
    logits,_ = model.apply({'params': params}, X, train=False, rngs={'dropout': rng})
    # jax.debug.print("{}",logits)
    entropy = get_uncertainty_score(logits,temperature=0.5)
    # jax.debug.print("ENTROPY: {}",entropy)
    # preds = nn.sigmoid(logits)
    # loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    """"""


    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    
    g_y = jac_map(params, model, X, rng)
    g_y_2 = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])

    # jax.debug.print('{}',[g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])
    dir = K-X
    # jax.debug.print("DIR: {}",dir)
    # jax.debug.print("DIR SHAPE: {}",dir.shape)
    # jax.debug.print("Nans in DIR: {}",jnp.isnan(dir).any())
    dir_norm = dir / jnp.linalg.norm(dir, axis=1, keepdims=True)
    # K_norm = K / jnp.linalg.norm(K, axis=1, keepdims=True)
    directional_derivative = jnp.sum(g_y_2 * dir_norm, axis=1)  # shape (N,)
    
    magnitude = jnp.linalg.norm(dir,axis=1)
    sign = jnp.tanh(20.0*directional_derivative)

    """
    """

    # EPS = 1e-8

    # map_dd = vmap(lambda a_row, b_col: 1 - (jnp.dot(a_row, b_col) /
    #                           (jnp.linalg.norm(a_row) * jnp.linalg.norm(b_col) + EPS)), in_axes=(0, 1))
    
    # dd = vmap(lambda K_slice: map_dd(K_slice+EPS, g_y_2.T+EPS), in_axes=1)(K)

    """
    """
    # loss = 1 + sign
    # jax.debug.print('dd: {}',directional_derivative)

    # jax.debug.print('e: {}',entropy)
    # jax.debug.print('dd/e: {}',directional_derivative/entropy)
    # loss = directional_derivative*entropy
    # loss = nn.softplus(directional_derivative) #/(entropy + 1e-6)
    # loss = nn.relu(directional_derivative)
    # jax.debug.print('loss: {}',loss)
    # jax.debug.print("Loss shape: {}",loss)
    # jax.debug.print("Loss shape: {}",entropy)
    # loss = jnp.dot(loss,entropy.T,)

    # loss = loss*(entropy/ jnp.max(entropy))

    # jax.debug.print("Loss shape: {}",loss)
    
    # loss = jnp.max(sign/magnitude,0)
    # loss = jnp.max(sign,0)
    # loss = nn.relu(directional_derivative/magnitude)
    # loss = nn.softplus(10*directional_derivative)/(magnitude)
    
    loss = nn.softplus(10*directional_derivative)/10

    
    # loss = nn.softplus(directional_derivative)/magnitude
    # jax.debug.print("X: {}",X)
    # jax.debug.print("K: {}",K)
    # jax.debug.print("Y: {}",Y)
    # jax.debug.print("GY: {}",g_y_2)
    
    # jax.debug.print("directional derivative: {}",directional_derivative)
    # # # jax.debug.print("magnitude: {}, Shape: {}",magnitude,magnitude.shape)
    # jax.debug.print("sign: {}",sign)
    # jax.debug.print("loss: {}",loss)
    
    return jnp.mean(loss)

def multiclass_direction_hessian(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector']
    
    # Helper to compute single-sample geometric properties
    def single_sample_loss(x, y, k):
        v = k - x
        v_norm = v / (jnp.linalg.norm(v) + 1e-7)
        
        # Define function: f(input) = logit of true class
        def f(input_x):
            # Assumes predict_wrapper handles single unbatched input
            return predict_wrapper(params, model, input_x, rng)[y]

        # 1. First Order: Directional Derivative (Slope)
        # JVP(f, x, v) -> grad(f). v
        val, slope = jax.jvp(f, (x,), (v_norm,))
        
        # 2. Second Order: Hessian Vector Product (Curvature)
        # We want to differentiate the gradient function along v
        def grad_dot_v(input_x):
            g = jax.grad(f)(input_x)
            return jnp.dot(g, v_norm)
        
        # JVP(grad_dot_v, x, v) -> v^T H v
        _, curvature = jax.jvp(grad_dot_v, (x,), (v_norm,))
        
        # 3. Combined Loss
        # Slope should be negative (Hinge)
        # Curvature should be zero (Linearity/Stability)
        slope_loss = nn.relu(slope + 0.1) # Margin 1.0
        curve_loss = jnp.abs(curvature)
        
        return slope_loss + 0.1 * curve_loss

    # Vmap over the batch
    losses = jax.vmap(single_sample_loss)(X, Y, K)
    
    return jnp.mean(losses)

def multiclass_direction_distance(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector'] 

    diff = K - X
    dist = jnp.linalg.norm(diff, axis=1)
    
    # 1. Compute gradients
    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    g_y = jac_map(params, model, X, rng)
    g_y_target = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])

    # 2. Directional Derivative (Raw, not normalized)
    # Using raw diff vector naturally scales derivative by distance
    dir_derivative = jnp.sum(g_y_target * diff, axis=1)

    # 3. Distance Weighting
    # Nearby points (low dist) -> High weight
    # Distant points (high dist) -> Low weight
    gamma = 0.5
    weights = 1.0 / (1.0 + gamma * dist)

    # 4. Energy Loss
    # We want negative derivative. Softplus(x) is monotonic increasing.
    # Minimizing Softplus(deriv) pushes deriv towards negative infinity.
    loss = weights * nn.softplus(dir_derivative)

    return jnp.mean(loss)

def multiclass_direction_hinge(params, model, batch, rng, config=None):
    X, Y, K = batch['X'], batch['Y'], batch['K']['vector'] 

    # 1. Calculate the vector from X to K
    diff = K - X
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
    direction = diff / dist  # Normalized direction vector v

    # 2. Compute gradients (Jacobian)
    jac_fn = jax.jacobian(predict_wrapper, argnums=2)
    jac_map = jax.vmap(jac_fn, in_axes=(None, None, 0, None), out_axes=0)
    
    g_y = jac_map(params, model, X, rng)
    # Extract gradient of the true class y
    g_y_target = jnp.array([g_y_i[y_i,:] for g_y_i,y_i in list(zip(g_y,Y))])

    # 3. Project gradient onto the counterfactual direction
    # This is the directional derivative (slope)
    slope = jnp.sum(g_y_target * direction.squeeze(), axis=1)

    # 4. Hinge Loss
    # We want slope < -margin (decreasing confidence)
    # Loss = ReLU(slope + margin)
    margin = 1.0 
    loss = nn.relu(slope + margin)
    
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
                        "K":{'vector':k_embs},
                        },rng)
    
    return (1 - alpha) * ce_loss + alpha * d_loss

def pad_to_max(seq, max_len):
        pad_len = max_len - seq.shape[0]
        print("SQEUENCE SHAPE:",seq.shape)
        pad = jnp.zeros((pad_len, seq.shape[1]))
        # pad = jnp.zeros((pad_len,) + seq.shape[-1])
        return jnp.concatenate([seq, pad], axis=0)


def multiclass_split_loss(params, model, batch, rng, alpha=1):
    
    X, Y, K = np.array(batch['X']), batch['Y'], batch['K']
    logits,_ = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
    
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))
    

    for key,val in K.items():
        K[key] =jnp.concat(val,axis = 1).squeeze()
        # K[key] = jnp.concatenate(val)
        # K[key] = jnp.stack([pad_to_max(jnp.array(k), k_count) for k in val]) 
    # print("concat SHAPE: ",K['X'].shape)
    # # # k_vectors_flat = K['K'].reshape(batch_size * k_count, *K['K'].shape[2:])
    # # print("K VECS FLAT SHAPE:",jnp.array(k_vectors_flat).shape)
    # # print(k_vectors_flat[0])
    
    
    d_loss = multiclass_direction(params,model,
                       {"X":K['X'],
                        'Y':K['Y'],
                        "K":{'vector':K['K']},
                        },rng)
    
    return (1 - alpha) * ce_loss + alpha * d_loss

def multiclass_split_loss_embedding(params, model, batch, rng, alpha=1):
    
    X, Y, K = np.array(batch['X']), batch['Y'], batch['K']
    logits, embeddings = model.apply({'params': params}, X, train=True, rngs={'dropout': rng})
   
    ce_loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=np.array(Y)))

    for key,val in K.items():
        
        if key=='text':
            continue
        # print(key)
        # print([len(v) for v in val])
        # print('dim 0:',type(val))
        # print('dim 1:',type(val[0]))

        K[key] = jnp.concatenate(val)
        # print(np.shape(K[key]))
        # K[key] = jnp.stack([pad_to_max(jnp.array(k), k_count) for k in val]) 
    # print("concat SHAPE: ",K['X'].shape)
    # # # k_vectors_flat = K['K'].reshape(batch_size * k_count, *K['K'].shape[2:])
    # # print("K VECS FLAT SHAPE:",jnp.array(k_vectors_flat).shape)
    # # print(k_vectors_flat[0])
    
    _, x_embs = model.apply({'params': params}, K['X'], train=False, rngs={'dropout': rng})
    _, k_embs = model.apply({'params': params}, K['K'], train=False, rngs={'dropout': rng})
    
    # jax.debug.print("K_EMBS: {}",k_embs.shape)
    # embedding_length = embeddings.shape[-1]
    # embeddings_expanded = jnp.expand_dims(embeddings, axis=1)        # (batch, 1, vector_length)
    # embeddings_expanded = jnp.repeat(embeddings_expanded, k_count, axis=1)  # (batch, n_cf, vector_length
    # embeddings_expanded = embeddings_expanded.reshape(batch_size * k_count, embedding_length)
    
    # y_expanded = jnp.expand_dims(np.array(Y), axis=1)
    # y_expanded = jnp.repeat(y_expanded, k_count, axis=1)  # (batch, n_cf, vector_length)
    # y_expanded = y_expanded.reshape(batch_size * k_count)
    
    # k_direction = batch_unit_vector(embeddings_expanded,k_embs)
    # k_direction = k_direction[:,jnp.newaxis,:]
    params_linear = {'linear1': params['linear1']}
    
    embedding_only = MulticlassEmbeddingOnlyModel(num_classes=logits.shape[-1])

    d_loss = multiclass_direction(params_linear,embedding_only,
                       {"X":x_embs,
                        'Y':K['Y'],
                        "K":{'vector':k_embs},
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


loss_functions =   {'direction':direction,
                    'combined_loss':combined_loss,
                    'multiclass_combined_loss':multiclass_combined_loss,
                    'multiclass_split_loss':multiclass_split_loss,
                    'combined_loss_embedding':combined_loss_embedding,
                    'multiclass_combined_loss_embedding':multiclass_combined_loss_embedding,
                    'multiclass_split_loss_embedding':multiclass_split_loss_embedding,
                    'multiclass_allcombined_loss_embedding':multiclass_allcombined_loss_embedding,
                    'combined_loss_imdb':combined_loss_imdb,
                    'cross_entropy':cross_entropy,
                    'multiclass_cross_entropy':multiclass_cross_entropy,
                    'cross_entropy_batch':cross_entropy_batch,
                    'cross_entropy_l2':cross_entropy_l2,
                    'gradient_supervision':gradient_supervision,
                    'gradient_supervision_embedding':gradient_supervision_embedding,
                    'multiclass_gradient_supervision':multiclass_gradient_supervision,
                    'multiclass_gradient_supervision_embedding':multiclass_gradient_supervision_embedding,
                    'multiclass_split_gs_embedding':multiclass_split_gs_embedding,
                    'gradient_supervision_basic':gradient_supervision_basic,
                    'direction_interactive': direction_interactive,
                    'direction_interactive_vectorized': direction_interactive_vectorized}