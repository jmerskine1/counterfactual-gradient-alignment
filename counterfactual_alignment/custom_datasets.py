from warnings import WarningMessage

import numpy as np
import pandas as pd
import pickle
import jax.numpy as jnp
import torch.utils.data as data
from counterfactual_alignment.utilities import visualise_classes, expand_data, gen_knowledge, boundary_filter, jagged_lists_to_array, convert_to_list_of_lists
from scipy.stats import multivariate_normal as mvn
from sklearn import datasets as sk_datasets



class customDataset(data.Dataset):

  def __init__(self,data):
    """
    Inputs:
        size  - Number of data points we want to generate (musn't exceed max datapoints in dataset)
        knowledge: whether to include counterfactual pairs
        seed  - The seed to use to create the PRNG state with which we want to generate the data points
    """
    
    self.X = data['X']
    self.Y = data['Y']
    self.K = data['K']
    
    self.K['vector'] = jagged_lists_to_array(convert_to_list_of_lists(self.K['vector']))
    
    
    
  def drop(self, idx):
    self.X             = {key:np.delete(self.X[key],idx,axis=0) for key in self.X}
    self.Y             = np.delete(self.Y,idx)
    
    # for i in range(len(self.K['vector'][idx])):
    self.K             = {key:np.delete(self.K[key],idx, axis=0) for key in self.K}
    # self.K             = {key:np.delete(self.K[key],[idx,i], axis=0) for key in self.K}
    # for key in self.K:
    #   del self.K[key][idx]
    

  def __getitem__(self, idx):
    
    X = {key:self.X[key][idx] for key in self.X}
    Y = self.Y[idx]
    K = {key:self.K[key][idx] for key in self.K}



    # return (self.data.X[idx],
    #         self.data.Y[idx],
    #         self.data.K['vector'][idx],
    #         self.data.K['label'][idx],
    #         self.data.K['magnitude'][idx])
    return {'X':X,'Y':Y,'K':K} #,'knowledge': K}

  def __len__(self):
    return len(self.Y)


class XOR():
                #self, x, y, variance = 1, covariance='default', n_vec=3, n_samples=10, max_delta=1.0
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """        
        self.X = {'vector': np.array(rng.randint(low=0, high=2, size=(size, 2)).astype(np.float32))}
        self.Y = np.array((self.X['vector'].sum(axis=1) == 1).astype(np.int32))
        self.X['vector'] += rng.normal(loc=0.0, scale=0.1, size=self.X['vector'].shape) # Add gaussian noise
        self.K = {'vector':None}

        # self.K = np.empty_like(self.Y)
    
    def optimum_classifier(self, z, probabilities=True):
        """
        Inputs  | z:      x,y coordinates of data to be classified.
        Outputs | probs:  array of probabilities for each class for input data.
        """
        probs = np.empty(0)
        for p in z:
            if p[0] <= 0.5 and p[1] <= 0.5:
                probs = jnp.append(probs,0)
            elif p[0] > 0.5 and p[1] > 0.5:
                probs = jnp.append(probs,0)
            elif p[0] < 0.5 and p[1] >= 0.5:
                probs = jnp.append(probs,1)
            elif p[0] >= 0.5 and p[1] < 0.5:
                probs = jnp.append(probs,1)

        probs = np.array([probs,abs(probs-1)])

        if not probabilities:
            probs = np.round(probs).astype(np.int32)
        
        self.probs = probs

        return np.atleast_1d(np.argmax(probs,axis=0))
    


class Gaussian():
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """
        self.num_classes = 2
        self.class_means = [[1,1],[-1,-1]]
        self.covariances = [np.eye(2),np.eye(2)]

        # class_sizes = np.zeros(num_classes)
        base       = size // self.num_classes
        leftover    = size  % self.num_classes

        class_sizes = list(np.append([base]*(self.num_classes-1),[base+leftover]))
        
        self.X = {'vector':np.concatenate([rng.multivariate_normal(np.array(self.class_means[i]), 
                                    self.covariances[i], class_sizes[i]) for i in range(self.num_classes)])}
        self.Y = np.concatenate([[i]*class_sizes[i] for i in range(self.num_classes)])

        self.K = {}
        # self.K = np.empty_like(self.Y)


    def optimum_classifier(self, z, probabilities=True):
        """
        Inputs  | z:      x,y coordinates of data to be classified.
        Outputs | probs:  array of probabilities for each class for input data.
        """
        pdfs = []

        for i in range(self.num_classes):
            pdfs.append(mvn.pdf(z, self.class_means[i], self.covariances[i]))

        probs = jnp.array([class_probs/sum(pdfs) for class_probs in pdfs])
        
        if not probabilities:
            probs = jnp.round(probs).astype(jnp.int32)
        
        self.probs = probs

        return jnp.atleast_1d(jnp.argmax(probs,axis=0))


class TwoMoons():
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """
        self.X, self.Y = sk_datasets.make_moons(size,random_state=rng)
        self.K = {}
        # self.K = np.empty_like(self.Y)


    def optimum_classifier(self, z, probabilities=True):
        """
        Inputs  | z:      x,y coordinates of data to be classified.
        Outputs | probs:  array of probabilities for each class for input data.
        """
        return None


class Circles():
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """
        self.X, self.Y = sk_datasets.make_circles(size,random_state=rng)
        self.K = {}
        # self.K = np.empty_like(self.Y)


    def optimum_classifier(self, z, probabilities=True):
        """
        Inputs  | z:      x,y coordinates of data to be classified.
        Outputs | probs:  array of probabilities for each class for input data.
        """
        return None

datasets = {'Gaussian':Gaussian,'XOR':XOR, 'TwoMoons':TwoMoons, 'Circles':Circles}

class genCustomDataset(data.Dataset):

  def __init__(self, dataset, size, knowledge_func=None, train=False, visualise=False, seed = 42, n_vec = 3):
    """
    Inputs:
        size  - Number of data points we want to generate (musn't exceed max datapoints in dataset)
        seed  - The seed to use to create the PRNG state with which we want to generate the data points
        d     - The centroid of each cluster, [+d,+d] for cluster 1, [-d,-d] for cluster 2
        gamma - Covariance of X1,X2
        direction_scheme = random, best_cf
    """
    self.size=size

    if not train:
        seed = seed + 1
    
    self.rng =  np.random.RandomState(seed)
    self.knowledge_func = knowledge_func
    self.visualise = visualise
    self.data = dataset(self.rng,self.size)
    self.X = self.data.X
    self.Y = self.data.Y
    self.K = self.data.K
    self.n_vec = n_vec

    if knowledge_func != None and train:
        gen_knowledge(self,knowledge_func=self.knowledge_func)
    elif knowledge_func == None and train:
        print("Warning: Training data with no knowledge function.")

    if visualise:
        visualise_classes(self.data,knowledge=bool(knowledge_func))


  def drop(self, idx):
    self.data.X['vector']             = np.delete(self.data.X['vector'],idx,axis=0)
    self.data.Y             = np.delete(self.data.Y,idx)
    self.data.K['vector']   = np.delete(self.data.K['vector'],idx,axis=0)
    self.data.K['label']    = np.delete(self.data.K['label'],idx)
    self.data.K['magnitude']= np.delete(self.data.K['magnitude'],idx)

  def __getitem__(self, idx):
    X = {'vector':self.data.X['vector'][idx]}
    Y = self.data.Y[idx]
    try:
      K = {key:self.data.K[key][idx] for key in self.data.K}
    except:
      K = {key:None for key in self.data.K}

    # return (self.data.X[idx],
    #         self.data.Y[idx],
    #         self.data.K['vector'][idx],
    #         self.data.K['label'][idx],
    #         self.data.K['magnitude'][idx])
    return {'X':X,'Y':Y,'K':K} #,'knowledge': K}

  def __len__(self):
    return len(self.data.Y)

"""
####################################################################################################################################
ADULT DATASET INCOMPLETE
(uncomment whole block)
####################################################################################################################################
"""

# class Adult(data.Dataset):
#   # for now, just working with age & hours per week, splitting by </> 50K
#   def __init__(self, size, seed, train=True, direction_scheme = default_direction_scheme, visualise = False, x0 = 'education-num', x1 = 'hours-per-week',n_vec=3, n_samples=10, max_delta=0.5):
#     """
#     Inputs:
#         size  - Number of data points we want to generate
#         seed  - The seed to use to create the PRNG state with which we want to generate the data points
#         direction_scheme = hyperparams, best_cf

#         dataset classes: >50K, <=50K.
#         x, y:      any two from self.names  ###(for now, needs to be continuous (or one-hot encoded?))
#     """
#     self.names = ['age',             # : continuous.
#                   'workclass',       # : Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
#                   'fnlwgt',          # : continuous.
#                   'education',       # : Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
#                   'education-num',   # : continuous.
#                   'marital-status',  # : Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
#                   'occupation',      # : Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
#                   'relationship',    # : Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
#                   'race',            # : White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
#                   'sex',             # : Female, Male.
#                   'capital-gain',    # : continuous.
#                   'capital-loss',    # : continuous.
#                   'hours-per-week',  # : continuous.
#                   'native-country',  # : United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
#                   'label']           # : '<=50K','>50K'           
    
#     self.size=size
#     self.direction_scheme = direction_scheme
#     self.visualise = visualise

#     # Create function which reads dataset from adult.data and adult.test
#     suffix = '.data'
#     if not train:
#         suffix = '.test'
    
#     adult_data = 'data/adult/adult' + suffix
#     data = pd.read_csv(adult_data,names=self.names,index_col=False,na_values='?')

#     data = data.dropna()
#     labels = data['label'].values
    
#         # Create function which randomly (using PRNG key) selects indices from total dataset
#     if self.size > len(data):
#         WarningMessage(["Requested size (" + str(self.size) + ") exceeds total number of points in dataset (" + str(len(data)) + "). Limiting to " + str(len(data)) + " points."])
#         self.size = len(data)

#     rng = np.random.default_rng(seed)
#     rints = rng.integers(low=0, high=len(data), size=self.size) 
        
#     self.data = jnp.array(np.column_stack((data[x0].values[rints],data[x1].values[rints])))
    
#     condition1 = np.unique(labels)[0]
#     self.label = jnp.array([0 if x ==  condition1 else 1 for x in labels[rints]])    # One-hot encode 0 = <=50k, 1 = >50K
    
#     if self.direction_scheme == 'random':
#         self.directions, self.direction_label, self.direction_distance, self.indices = random_directions(
#             self.data, self.label, self.optimum_classifier, n_vec, n_samples, max_delta)
#     elif self.direction_scheme == 'best_cf':
#         self.directions, self.direction_label, self.direction_distance, self.indices = best_direction(
#             self.data, self.optimum_classifier)
#     else:
#         ValueError("'directions' must be either 'random' or 'best_cf'")
    
#   def optimum_classifier(self, z, probabilities=True):
#     # load? model
#     # logits = state.apply_fn(params, data_input).squeeze(axis=-1)
#     # pred_labels = (logits > 0).astype(jnp.float32)

#     probs = np.ones(np.shape(z)[0])
#     probs[:] = 0.5
#     # probs[:] =  test # model.predict(z) output(s)
    
#     if not probabilities:
#         probs = jnp.round(probs).astype(jnp.int32)

#     return probs

  
#   def __getitem__(self, idx):
#     direction = self.directions[idx]
#     direction_label = self.direction_label[idx]
#     direction_distance = self.direction_distance[idx]
#     point_idx = self.indices[idx]
#     data_point = self.data[point_idx]
#     data_label = self.label[point_idx]
#     # return x, y, direction, direction_label
#     return data_point, data_label, direction, direction_label, direction_distance

#   def __len__(self):
#     return self.size


