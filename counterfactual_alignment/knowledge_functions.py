
import numpy as np
import jax.numpy as jnp
from tqdm.auto import tqdm
import random
from sklearn.neighbors import NearestNeighbors
import networkx as nx
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
    Warning("Counterfactual function currently ignores n_vec and just generates one vector")
    subsample=False

    if n_vec < 1:
        subsample=True
        rng =np.random.default_rng()
        K_i = rng.choice(len(X),size=int(n_vec*len(X)))
        n_vec = 1

        

    cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function = classifier,
                                                      dataset = X,
                                                      categorical_indices=[],
                                                      default_numerical_step_size=0.1)
    
    directions = np.zeros((len(X),n_vec, 2))
    
    direction_label = np.zeros((len(X),n_vec))
    direction_distance = np.zeros((len(X),n_vec))

    for i, x in tqdm(enumerate(X)):
        if subsample:
            if i in K_i:
                dir, distance = gen_best_vec(x,cf_explainer)
                
                # single label version
                directions[i,0, :] = dir
                direction_label[i,0] = -1
                direction_distance[i,0] = distance


        else:
            dir, distance = gen_best_vec(x,cf_explainer)
            
            # single label version
            directions[i,0, :] = dir
            direction_label[i,0] = -1
            direction_distance[i,0] = distance
    
    K = {'vector'    :directions, 
         'label'     :direction_label,
         'magnitude' :direction_distance}
    
    return K




def counterfactual(X, classifier, dims = 2, n_vec=3, n_samples=10, max_delta=1.0):
    print(f"Generating {n_vec} counterfactual directions...")
  # Lets generate the nearest point where the classification boundary changes
    # For now, generate the nearest n_vec counterfactuals, and keep the rest the same?
    Warning("Counterfactual function currently ignores n_vec and just generates one vector")
    subsample=False

    if n_vec < 1:
        subsample=True
        rng =np.random.default_rng()
        K_i = rng.choice(len(X),size=int(n_vec*len(X)))
        n_vec = 1

        

    cf_explainer = fatf_cf.CounterfactualExplainer(predictive_function = classifier,
                                                      dataset = X,
                                                      categorical_indices=[],
                                                      default_numerical_step_size=0.1)
    
    directions = np.zeros((len(X),n_vec, 2))
    
    direction_label = np.zeros((len(X),n_vec))
    direction_distance = np.zeros((len(X),n_vec))

    for i, x in tqdm(enumerate(X)):
        if subsample:
            if i in K_i:
                dir, distance = gen_best_vec(x,cf_explainer)
                
                # single label version
                directions[i,0, :] = dir
                direction_label[i,0] = -1
                direction_distance[i,0] = distance


        else:
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
    
    directions =  np.zeros((len(X), len(X[0])))
    distances =  np.zeros((len(X), len(X[0])))
    
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

def counterfactual_feasible_vector_paths(X, Y, classifier, n_samples=3):
    print(f"Generating {n_samples} counterfactual samples per observation ...")
    classes = list(set(Y))

    cf_explainer = fatf_cf.CounterfactualExplainer(
        predictive_function=classifier,
        dataset=X,
        numerical_indices=[0, 1],
        default_numerical_step_size=0.1,
        max_counterfactual_length=1
    )

    origins = np.zeros((len(X), n_samples, len(X[0])))
    directions = distances = np.zeros((len(X), len(X[0])))

    for i, x in tqdm(enumerate(X)):
        print('generating :', i)
        counterfactual_class = int(next(c for c in classes if c != Y[i]))

        cf, distance, label = cf_explainer.explain_instance(
            np.array(x),
            counterfactual_class,
            normalise_distance=False
        )

        if np.shape(cf)[0] > 1:
            cf = [list(cf[0])]

        cf_point = np.array(cf[0])
        x = np.array(x)

        if np.allclose(cf_point, x):
            directions[i] = np.array([np.nan, np.nan])
            continue

        # Interpolate along x and compute y = x^2 to follow true distribution
        x_start, x_end = sorted([x[0], cf_point[0]])
        boundary = abs(x_end-x_start) * 1/(1+n_samples)
        
        x_vals = np.linspace(x_start+boundary, x_end-boundary, n_samples)
        
        
        for k, x_val in enumerate(x_vals):
            y_val = x_val ** 2
            origins[i, k, :] = [x_val, y_val]

        # Vector from original x to cf (used for directional derivatives later)
        directions[i], _ = get_unit_vec(x, cf_point)

    return {
        'origin': origins,
        'vector': directions
    }


import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import heapq

def knn_boundary_points(X_start, classifier, Z, n_breadcrumbs, k=10):
    Z_aug = np.vstack([Z, X_start])
    start_idx = len(Z_aug) - 1

    labels = classifier(Z_aug)
    start_label = labels[start_idx]

    nbrs = NearestNeighbors(n_neighbors=k).fit(Z_aug)
    distances, indices = nbrs.kneighbors(Z_aug)

    G = nx.Graph()
    for i in range(len(Z_aug)):
        for j, d in zip(indices[i], distances[i]):
            if i != j:
                G.add_edge(i, j, weight=d)

    # Dijkstra expansion
    visited = set()
    pq = [(0, start_idx, [start_idx])]

    while pq:
        dist, node, path = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)

        # STOP AT FIRST CLASS FLIP
        if labels[node] != start_label:
            path_points = Z_aug[path]

            idxs = np.linspace(
                0, len(path_points) - 1,
                n_breadcrumbs + 1,
                dtype=int
            )

            return path_points[idxs[:-1]]

        for neigh in G.neighbors(node):
            if neigh not in visited:
                w = G[node][neigh]["weight"]
                heapq.heappush(pq, (dist + w, neigh, path + [neigh]))

    raise RuntimeError("No boundary found")

# def counterfactual_breadcrumbs(X_start,X_end, Z, classifier, n_breadcrumbs=3, k = 10):
    
#     # boundary_0 = X_end[0]-X_start[0] 
#     # boundary_1 = (0.2/(1+n_breadcrumbs))
#     # print("B0: ",boundary_0)
#     # print("B1: ",boundary_1)
#     boundary = (X_end[0]-X_start[0]) * (0.5/(1+n_breadcrumbs))
#     x = np.linspace(X_start[0]+boundary, X_end[0]-boundary, n_breadcrumbs)
    
#     y = x**2
#     x = x + random.normalvariate(0,0.05)
#     y = y + random.normalvariate(0,0.05)
    
#     origins = np.zeros((n_breadcrumbs,len(X_start)))
#     vectors = np.zeros_like(origins)
    
#     for b in range(n_breadcrumbs):
#         if b < n_breadcrumbs-1:
#             next_vec = [x[b+1],y[b+1]]
#         else:
#             next_vec = X_end
        
#         origins[b] = (x[b],y[b])
        
#         vectors[b],distance = get_unit_vec(origins[b],next_vec)

#     return {'origins':origins,
#             'vectors': vectors,
#             'labels': classifier(origins)}
# def counterfactual_breadcrumbs(
#     X_start,
#     classifier,
#     Z,
#     n_breadcrumbs=3,
#     k=10
# ):
#     points = knn_boundary_points(
#         X_start,
#         classifier,
#         Z,
#         n_breadcrumbs,
#         k=k
#     )

#     origins = np.zeros((len(points), len(X_start)))
#     vectors = np.zeros_like(origins)

#     for b in range(len(points)):
#         next_vec = points[b + 1] if b < len(points) - 1 else points[b]
#         origins[b] = points[b]
#         vectors[b], _ = get_unit_vec(origins[b], next_vec)

#     return {
#         'origins': origins,
#         'vectors': vectors,
#         'labels': classifier(origins)
#     }

import numpy as np
import heapq
from sklearn.neighbors import NearestNeighbors


def get_unit_vec(a, b):
    v = b - a
    n = np.linalg.norm(v)
    if n == 0:
        return np.zeros_like(v), 0.0
    return v / n, n


def find_boundary_segment(X_start, classifier, Z, k=10):
    """
    Returns (x_same, x_flip):
    x_same -> last point with same class as X_start
    x_flip -> first neighboring point with different class
    """

    Z = np.vstack([Z, X_start])
    start_idx = len(Z) - 1

    labels = classifier(Z)
    start_label = labels[start_idx]

    nbrs = NearestNeighbors(n_neighbors=k).fit(Z)
    distances, indices = nbrs.kneighbors(Z)

    visited = set()
    pq = [(0.0, start_idx)]

    while pq:
        cost, node = heapq.heappop(pq)

        if node in visited:
            continue
        visited.add(node)

        for neigh, d in zip(indices[node], distances[node]):
            if neigh in visited:
                continue

            if labels[neigh] != start_label:
                return Z[node], Z[neigh]

            heapq.heappush(pq, (cost + d, neigh))

    raise RuntimeError("No class boundary found")


def boundary_bisection(
    x_same,
    x_flip,
    classifier,
    max_iter=30,
    tol=1e-6
):
    """
    Refines boundary location between x_same and x_flip
    """

    y0 = classifier(x_same.reshape(1, -1))[0]

    a = x_same.copy()
    b = x_flip.copy()

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        y_mid = classifier(mid.reshape(1, -1))[0]

        if y_mid == y0:
            a = mid
        else:
            b = mid

        if np.linalg.norm(b - a) < tol:
            break

    return 0.5 * (a + b)


def sample_towards_boundary(x_start, x_boundary, n_breadcrumbs):
    """
    Regularly spaced samples between x_start and boundary
    """
    t = np.linspace(0, 1, n_breadcrumbs + 1, endpoint=False)[1:]
    return (1 - t)[:, None] * x_start + t[:, None] * x_boundary


def counterfactual_breadcrumbs(
    X_start,
    classifier,
    Z,
    n_breadcrumbs=3,
    k=50
):
    """
    Main entry point
    """

    # 1. Locate nearest boundary edge
    x_same, x_flip = find_boundary_segment(
        X_start=X_start,
        classifier=classifier,
        Z=Z,
        k=k
    )

    # 2. Refine exact boundary via bisection
    x_boundary = boundary_bisection(
        x_same,
        x_flip,
        classifier
    )

    # 3. Regular sampling toward boundary
    points = sample_towards_boundary(
        X_start,
        x_boundary,
        n_breadcrumbs
    )

    origins = points
    vectors = np.zeros_like(origins)

    for i in range(n_breadcrumbs):
        next_pt = points[i + 1] if i < n_breadcrumbs - 1 else x_boundary
        vectors[i], _ = get_unit_vec(origins[i], next_pt)

    return {
        "origins": origins,
        "vectors": vectors,
        "labels": classifier(origins)
    }


def counterfactual_feasible_moon_paths(X, Y, classifier, n_samples=3):
    print(f"Generating {n_samples} counterfactual samples per observation ...")
    classes = list(set(Y))

    cf_explainer = fatf_cf.CounterfactualExplainer(
        predictive_function=classifier,
        dataset=X,
        numerical_indices=[0, 1],
        default_numerical_step_size=0.1,
        max_counterfactual_length=1
    )

    origins = np.zeros((len(X), n_samples, len(X[0])))
    directions = distances = np.zeros((len(X), len(X[0])))

    for i, x in tqdm(enumerate(X)):
        counterfactual_class = int(next(c for c in classes if c != Y[i]))

        cf, distance, label = cf_explainer.explain_instance(
            np.array(x),
            counterfactual_class,
            normalise_distance=False
        )

        if np.shape(cf)[0] > 1:
            cf = [list(cf[0])]

        cf_point = np.array(cf[0])
        x = np.array(x)

        if np.allclose(cf_point, x):
            directions[i] = np.array([np.nan, np.nan])
            continue

        # Interpolate along x and compute y = x^2 to follow true distribution
        x_start, x_end = sorted([x[0], cf_point[0]])
        boundary = abs(x_end-x_start) * 0.05
        
        x_vals = np.linspace(x_start+boundary, x_end-boundary, n_samples)
        
        
        for k, x_val in enumerate(x_vals):
            y_val = x_val ** 2
            origins[i, k, :] = [x_val, y_val]

        # Vector from original x to cf (used for directional derivatives later)
        directions[i], _ = get_unit_vec(x, cf_point)

    return {
        'origin': origins,
        'vector': directions
    }

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
             