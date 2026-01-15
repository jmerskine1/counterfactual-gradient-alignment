from warnings import WarningMessage

import numpy as np
import pandas as pd
import pickle
import jax.numpy as jnp
import torch.utils.data as data
from counterfactual_alignment.utilities import visualise_classes, expand_data, boundary_filter, jagged_lists_to_array, convert_to_list_of_lists
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

        try:
            self.text = data['text']
        except:
            pass

        self.X = data['X']
        self.Y = data['Y']
        self.K = data['K']

        if 'vector' in self.K and self.K['vector'] is not None:
            self.K['vector'] = jagged_lists_to_array(convert_to_list_of_lists(self.K['vector']))
        elif 'K' in self.K and self.K['K'] is not None:
            pass



    def drop(self, idx):
        try:
            self.text = np.delete(self.text,idx)
        except:
            pass
        # self.text          = np.delete(self.text,idx)
        self.X             = np.delete(self.X,idx)
        self.Y             = np.delete(self.Y,idx)

        # for i in range(len(self.K['vector'][idx])):
        self.K             = {key:np.delete(self.K[key],idx, axis=0) for key in self.K}
        # self.K             = {key:np.delete(self.K[key],[idx,i], axis=0) for key in self.K}
        # for key in self.K:
        #   del self.K[key][idx]


    def __getitem__(self, idx):

        # X = {key:self.X[key][idx] for key in self.X}
        try:
            text = self.text[idx]
        except:
            text = False
        X = self.X[idx]
        Y = self.Y[idx]
        K = {key:self.K[key][idx] for key in self.K}



        # return (self.data.X[idx],
        #         self.data.Y[idx],
        #         self.data.K['vector'][idx],
        #         self.data.K['label'][idx],
        #         self.data.K['magnitude'][idx])
        if text:
            return {'text':text,'X':X,'Y':Y,'K':K} #,'knowledge': K}
        else:
            return {'X':X,'Y':Y,'K':K}

    def __len__(self):
        return len(self.Y)

    def subset(self, indices):
        """
        Return a deep copy of the dataset containing only the samples
        corresponding to the given list/array of indices.
        """
        import copy
        indices = np.array(indices)

        # Handle text field if present
        try:
            subset_text = np.array(self.text)[indices]
        except AttributeError:
            subset_text = None
        except Exception:
            subset_text = None

        subset_X = np.array(self.X)[indices]
        subset_Y = np.array(self.Y)[indices]

        # Copy K dictionary fields safely
        subset_K = {}
        for key in self.K:
            # arr = np.array(self.K[key])
            arr = self.K[key]
            # Some K fields might be None or jagged
            if arr is not None and len(arr) > 0:
                # subset_K[key] = arr[indices]
                subset_K[key] = [arr[i] for i in indices]
            else:
                subset_K[key] = arr

        # Build new dataset dict
        data_subset = {
            'text': subset_text if subset_text is not None else [],
            'X': subset_X,
            'Y': subset_Y,
            'K': subset_K
        }

        # Return a new dataset instance
        return customDataset(copy.deepcopy(data_subset))
    
    def combine(self, other):
        """
        Combine this dataset with another customDataset instance.
        Returns a new customDataset containing data from both.
        """

        import copy
        assert isinstance(other, customDataset), "Can only combine with another customDataset"

        # Combine text field if available
        try:
            if hasattr(self, "text") and hasattr(other, "text"):
                combined_text = np.concatenate([np.array(self.text), np.array(other.text)], axis=0)
            elif hasattr(self, "text"):
                combined_text = np.array(self.text)
            elif hasattr(other, "text"):
                combined_text = np.array(other.text)
            else:
                combined_text = []
        except Exception:
            combined_text = []

        # Combine X and Y
        combined_X = np.concatenate([np.array(self.X), np.array(other.X)], axis=0)
        combined_Y = np.concatenate([np.array(self.Y), np.array(other.Y)], axis=0)

        # Combine K dictionary fields
        combined_K = {}
        all_keys = set(self.K.keys()) | set(other.K.keys())
        for key in all_keys:
            arr1 = self.K.get(key, [])
            arr2 = other.K.get(key, [])
            
            if len(arr1) == 0:
                combined_K[key] = arr2
            elif len(arr2) == 0:
                combined_K[key] = arr1
            else:
                # Check if they are lists or numpy arrays
                if isinstance(arr1, list) and isinstance(arr2, list):
                    combined_K[key] = arr1 + arr2
                elif isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
                    combined_K[key] = np.concatenate([arr1, arr2], axis=0)
                else:
                    # Try to convert to list and combine if types mismatch or other iterable
                    combined_K[key] = list(arr1) + list(arr2)

        # Build combined data dict
        combined_data = {
            'text': combined_text,
            'X': combined_X,
            'Y': combined_Y,
            'K': combined_K
        }

        # Return new dataset instance
        return customDataset(copy.deepcopy(combined_data))



class XSQUARED():
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """        

        self.X = np.array([(x,x**2) for x in np.linspace(-1,1,size)]) 
        self.X += rng.normal(loc=0.0, scale=0.1, size=self.X.shape) # Add gaussian noise# np.random.normal(rng,(size,2))*0.1 # multiply by scaling facto
        self.Y = np.zeros_like(self.X[:,0])
        self.Y[int(size/2 + size%2):] = 1
        self.K = None

        
    def optimum_classifier(self,Z):
            """
            Inputs  | z:      x,y coordinates of data to be classified.
            Outputs | probs:  array of probabilities for each class for input data.
            """
            return np.array([int(z[0]>=0) for z in Z])


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
        
        self.X = np.array(rng.randint(low=0, high=2, size=(size, 2)).astype(np.float32))
        self.Y = np.array((self.X.sum(axis=1) == 1).astype(np.int32))
        self.X += rng.normal(loc=0.0, scale=0.1, size=self.X.shape) # Add gaussian noise
        
        self.K = None

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
        # print('SHAPE OF opt probs: ',np.shape(probs))
        # print('Example: ',probs)
        # print('SHAPE OF opt output: ',np.shape(np.atleast_1d(np.argmax(probs,axis=0))))
        # print('Example: ',np.atleast_1d(np.argmax(probs,axis=0)))
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
        
        self.X = np.concatenate([rng.multivariate_normal(np.array(self.class_means[i]), 
                                    self.covariances[i], class_sizes[i]) for i in range(self.num_classes)])
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

from sklearn.svm import SVC
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

        # 3. Train a near-optimal classifier
        self.clf = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
        self.clf.fit(self.X, self.Y)


    def optimum_classifier(self, z, probabilities=True):
        """
        Deterministic hardcoded classifier for the make_moons dataset.
        Takes a single input vector z = [x1, x2] and returns class 0 or 1.

        Parameters:
            z (np.ndarray): Input 2D point (shape: (2,)).

        Returns:
            int: Predicted label (0 or 1).
        """
        z = np.asarray(z)
        x1 = z[:,0]
        x2 = z[:,1]

        # Approximate lower moon: label = 0 if below boundary
        lower_moon = x1 < 1.0
        decision_boundary = 0.5 * np.sin(np.pi * x1)

        predictions = np.where(x2 > decision_boundary, 1, 0)
        # print('type1:',type(predictions))
        # print('shape1:',np.shape(predictions))
        # print('type2:',type(self.clf.predict(z)))
        # print('shape2:',np.shape(self.clf.predict(z)))
        # return predictions
        return self.clf.predict(z)
        


class Circles():
    def __init__(self, rng, size):
        """
        Inputs  | rng   : Pseudo-random number generator
                | size  : Size of dataset
        Outputs | X     : X,Y cooridnates of observations
                | Y     : Class label ([0,1...n])
                | K     : Knowledge - empty dict, for storing directional info
        """
        self.X, self.Y = sk_datasets.make_circles(size,random_state=rng,noise=0.01)
        self.K = {}
        outer  = [abs(np.sqrt(x[0]**2 + x[1]**2))for x in self.X[[idx for idx,val in enumerate(self.Y) if self.Y[idx] == 0]]]
        inner  = [abs(np.sqrt(x[0]**2 + x[1]**2))for x in self.X[[idx for idx,val in enumerate(self.Y) if self.Y[idx] == 1]]]
        self.decision_boundary = np.mean(inner) + (np.mean(outer) - np.mean(inner))/2 # radius halfway between max inner point and min outer point
        


    def optimum_classifier(self, z, probabilities=True):

        """
        Inputs  | z:      x,y coordinates of data to be classified.
        Outputs | probs:  array of probabilities for each class for input data.
        """
        # We know the decision boundary is r = 0.5 from (0,0)
        # for any point, if r >= 0.5, y = 0, else y = 1
        # print(z, [np.sqrt(z_i[0]**2 + z_i[1]**2)for z_i in z] ,np.array([int(np.sqrt(z_i[0]**2 + z_i[1]**2) <= 0.5) for z_i in z]))
        return np.array([int(abs(np.sqrt(z_i[0]**2 + z_i[1]**2)) <= self.decision_boundary) for z_i in z])

datasets = {'Gaussian':Gaussian,'XOR':XOR, 'TwoMoons':TwoMoons, 'Circles':Circles, "XSQUARED":XSQUARED}

class genCustomDataset(data.Dataset):

  def __init__(self, dataset, size, num_classes = None, knowledge_func=None, train=False, visualise=False, seed = 42, n_vec = 3):
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
    
    if num_classes:
       self.data = dataset(self.rng,self.size,num_classes=num_classes)
    else:
        self.data = dataset(self.rng,self.size)
    self.X = self.data.X
    self.Y = self.data.Y
    self.K = self.data.K
    self.n_vec = n_vec

    if knowledge_func != None and train:
        self.K = self.knowledge_func(self.X,self.data.optimum_classifier,n_vec=self.n_vec) # to this(self,knowledge_func=self.knowledge_func)
        
    elif knowledge_func == None and train:
        print("Warning: Training data with no knowledge function.")

    if visualise:
        visualise_classes(self.data,knowledge=bool(knowledge_func))


  def drop(self, idx):
    # self.data.X['vector']             = np.delete(self.data.X['vector'],idx,axis=0)
    self.X             = np.delete(self.X,idx)
    self.Y             = np.delete(self.Y,idx)
    self.K['vector']   = np.delete(self.K['vector'],idx,axis=0)
    self.K['label']    = np.delete(self.K['label'],idx)
    self.K['magnitude']= np.delete(self.K['magnitude'],idx)

  def __getitem__(self, idx):
    X = self.X[idx] #{'vector':self.data.X['vector'][idx]}
    Y = self.Y[idx]
    try:
      K = {key:self.K[key][idx] for key in self.K}
    except:
      K = {key:None for key in self.K}

    # return (self.data.X[idx],
    #         self.data.Y[idx],
    #         self.data.K['vector'][idx],
    #         self.data.K['label'][idx],
    #         self.data.K['magnitude'][idx])
    return {'X':X,'Y':Y,'K':K} #,'knowledge': K}

  def __len__(self):
    return len(self.Y)

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


