
import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm

# FAT Forensics Counterfactual Explainer
import fatf.transparency.predictions.counterfactuals as fatf_cf
from counterfactual_alignment.utilities import get_rand_vec, get_unit_vec


def gen_best_vec(x,cf_explainer):      
  cfs = cf_explainer.explain_instance(np.array(x),normalise_distance=True)
  
  return get_unit_vec(x,cfs[0][0])


def random_vectors(dataset, dims = 2, n_vec=3, n_samples=10, max_delta=1.0):
    print(f"Generating {n_vec} random directions...")

    X = dataset.data.X['vector']
  # Lets generate some random directions, and whether the class can change
    # with that direction
    cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function = dataset.data.optimum_classifier,
                                                      dataset = X,
                                                      categorical_indices=[],
                                                      default_numerical_step_size=0.1)

    # For each point we'll generate n_vec random unit vectors around the points
    directions = np.zeros((len(X), n_vec, 2))
    direction_label = np.ones((len(X), n_vec))
    direction_distance = np.ones((len(X), n_vec))
    
    for i, x in tqdm(enumerate(X)):
        print(f'getting {n_vec} vectors for observation {i}')
        vecs =[get_rand_vec(dims) for _ in range(n_vec)]
        directions[i, :, :] = np.stack(vecs)
        print(f'directions for label {i}: {np.shape(directions)}')
        # For each direction, generate labels along a vector from point x
        for k, v in enumerate(vecs):
            # Sample along direction
            delta = np.linspace(0.0, max_delta, n_samples)
            xs = delta*np.repeat(jnp.expand_dims(v, 1), n_samples, axis=1) + \
            np.repeat(np.expand_dims(x, 1), n_samples, axis=1)

            predicted_class = dataset.data.optimum_classifier(xs.T, probabilities=False)
            # If the other class is along this line, assign label -1
            if any(predicted_class != dataset.data.Y[i]):
                direction_label[i, k] = -1 
        
        _, direction_distance[i,:] = gen_best_vec(x,cf_explainer)
    

    return directions, direction_label, direction_distance


def counterfactual_vector(X, classifier, dims = 2, n_vec=3, n_samples=10, max_delta=1.0):
    print(f"Generating {n_vec} counterfactual directions...")
  # Lets generate the nearest point where the classification boundary changes
    # For now, generate the nearest n_vec counterfactuals, and keep the rest the same?
    

    cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function = classifier,
                                                      dataset = X,
                                                      categorical_indices=[],
                                                      default_numerical_step_size=0.1)
    
    directions = np.zeros((len(X),n_vec, 2))
    
    direction_label = -np.ones((len(X),n_vec))
    direction_distance = np.ones((len(X),n_vec))

    for i, x in tqdm(enumerate(X)):

      dir, distance = gen_best_vec(x,cf_explainer)
      
      # single label version
      directions[i,0, :] = dir
      direction_label[i,0] = -1
      direction_distance[i,0] = distance
    
    K = {'vector'    :directions, 
         'label'     :direction_label,
         'magnitude' :direction_distance}
    
    return K


def counterfactual_vector_paths(X,Y, classifier, n_samples=3):
    print(f"Generating {n_samples} counterfactual samples per observation ...")
  # Lets generate the nearest point where the classification boundary changes
    # For now, generate the nearest n_vec counterfactuals, and keep the rest the same?
    # classes = np.unique(Y)
    classes = list(set(Y))

    cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function = classifier,
                                                      dataset = X,
                                                      numerical_indices=[0,1],
                                                      default_numerical_step_size=0.1,
                                                      max_counterfactual_length=1)
    
    origins = np.zeros((len(X),n_samples, len(X[0])))
    
    directions = distances =  np.zeros((len(X), len(X[0])))
    
    for i, x in tqdm(enumerate(X)):
      
      counterfactual_class = int(next(c for c in classes if c != Y[i]))
      
      cf, distance, label = cf_explainer.explain_instance(np.array(x),
                                                                counterfactual_class,
                                                                normalise_distance=False)
      
      if np.shape(cf)[0] > 1: # horrible fix for unknown behaviour where 2 cfs are generated instead of one - only seems to occur when x == cf
          cf = [list(cf[0])]

      if np.allclose(np.array(cf[0]),np.array(x)):
         directions[i] = np.array([np.nan,np.nan])
      else:  
        directions[i],_ = get_unit_vec(x,cf[0])
      
      vector = cf - x
      boundary = 0.1

      for k,d in enumerate(np.linspace(0.0+boundary, 1-boundary, n_samples)):
        origins[i,k,:] = x + d*vector
    
    return {'origin'    :origins,
            'vector'    :directions}

def interactive_vector(dataset, dims = 2, n_vec=3, n_samples=10, max_delta=1.0):
    n = len(dataset.data.X)
    directions = [np.empty((0,2)) for _ in range(n)]
    direction_label = [[-1] for _ in range(n)]
    direction_distance = [[] for _ in range(n)]
    # directions = np.empty((len(dataset.data.X),1,2))
    # direction_distance = np.empty((len(dataset.data.X),1))

    # # direction_label always -1? Yes for now
    # direction_label = np.empty((len(dataset.data.X),1))
    
    return directions, direction_label, direction_distance

knowledge_functions = {'random':random_vectors,
             'counterfactual':counterfactual_vector,
             'interactive':interactive_vector}
             