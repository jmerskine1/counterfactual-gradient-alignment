import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl


# Path to your CSV folder (adjust this to where your files are stored)
csv_folder = "data/mnist-in-csv"
n_train = 200

visualise = False
boundary_only = False
connected_paths = True
path_type = 'pca'  # 'raw' or 'pca'
path_length = 'direct'  # 'direct' or 'local'



def plot_paths(images, labels, indices):
    
    # 4. Plot them
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        plt.subplot(5, 5, i + 1)
        x = images[idx].reshape(28, 28)  # reshape to 28x28
        plt.imshow(x, cmap="gray")  # grayscale for MNIST
        plt.title(f"Index {idx}")

        plt.xticks([])   # remove tick labels, keep axis
        plt.yticks([])
        plt.xlabel(f"Label: {labels[idx]}")

    plt.tight_layout()
    plt.show()



def stratified_subsample(X, y, n_samples, random_state=42):
    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=n_samples,
        stratify=y,
        random_state=random_state
    )
    return X_sub, y_sub


def load_mnist_data(csv_path, n_samples=None):
    # Load MNIST from CSV (assuming first column is label, rest are pixels)
    df = pd.read_csv(csv_path, dtype=np.uint8)
    X = df.iloc[:, 1:]/255
    X = X.astype("float32")

    # X = df.iloc[:, 1:].values.astype(np.float32)
    
    y = df.iloc[:, 0].values  

    # Subsample if n_samples is specified
    if n_samples is not None:
        if n_samples > X.shape[0]:
            raise ValueError(f"n_samples {n_samples} exceeds dataset size {X.shape[0]}")

        X, y = stratified_subsample(X, y, n_samples)
    
    return X, y




# List all CSV files in the folder
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]

# Read all CSVs into a dictionary of DataFrames
dataframes = {file: pd.read_csv(os.path.join(csv_folder, file)) for file in csv_files}

# Example: show basic info for each file
for name, df in dataframes.items():
    print(f"\n--- {name} ---")
    print(df.head())      # first 5 rows
    print(df.info())      # column details


# number_of_samples = 10000
X,y = load_mnist_data("data/mnist-in-csv/mnist_train.csv",
                           n_samples=10000)

# X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.2, random_state=42,shuffle=False)

X_test = dataframes['mnist_test.csv'].drop(columns=["label"])  # everything except the target
y_test = dataframes['mnist_test.csv']["label"]  # everything except the

X_test, X_dev, y_test, y_dev = train_test_split(X_test, y_test, test_size=0.5, random_state=42,shuffle=True)

# y_train_df = pd.DataFrame(index=X_train.index)
# y_train_df['y'] = y_train

y_df = pd.DataFrame(index=X.index)
y_df['y'] = y


raw_paths  = pd.read_csv("data/paths/mnist_paths_space-raw_k-20_paths-100000_datasize-10000.csv")
pca_paths  = pd.read_csv("data/paths/mnist_paths_space-pca_k-20_paths-100000_datasize-10000.csv")
face_paths = pd.read_csv("data/paths/mnist_paths_FACE_paths-70000_datasize-10000.csv")
face_paths['path_id'] = face_paths['path_id'].str.replace('_to', '', regex=False)

face_subset = face_paths.iloc[0:3]
plot_paths(np.array(X),list(y),list(face_subset['index'])+[508])

raw_demo_path = raw_paths.iloc[0:4]
pca_demo_path = pca_paths.iloc[0:8]
face_demo_path = face_paths.iloc[0:3]

raw_demo_path.head()

for demo_path in [raw_demo_path,pca_demo_path,face_demo_path]:
    indices = list(demo_path['index'])
    final_ind = list([demo_path['next_index'].iloc[-1]])
    
    plot_paths(np.array(X),list(y),indices+final_ind)

path_dict = {
    'raw': raw_paths,
    'pca': pca_paths,
    'face': face_paths
}


plt.figure(figsize=(8, 5))

all_lengths = [
    paths['path_id'].value_counts()
    for paths in path_dict.values()
]

labels = list(path_dict.keys())

plt.hist(all_lengths, 
         bins=range(1, max(max(l) for l in all_lengths) + 1),
         stacked=True,
         label=labels,
         align='left',
         rwidth=0.9)

plt.legend()
plt.xlabel("Path Length")
plt.ylabel("Count")
plt.title("Stacked Histogram of Path Lengths")
plt.show()


grouped_paths = face_paths.groupby('path_id')

import pandas as pd

def dedupe_variable_length_journeys(df, index_col="index", next_col="next_index"):
    """
    Deduplicate variable-length journeys defined by (index → next_index) chains.
    
    A journey is any contiguous chain where each row's `index` equals the
    previous row's `next_index`. Journeys can be any length.
    
    Returns only one copy of each unique journey sequence.
    Journeys with the same path but different sequences are all preserved.
    Journeys with identical sequences are collapsed into one.
    """
    
    # Ensure sorted by original order
    df = df.copy().sort_index()

    # 1. Identify where journeys break
    df["new_journey"] = df[index_col] != df[next_col].shift(1)
    df["journey_id_real"] = df["new_journey"].cumsum()

    # 2. Build canonical sequence signatures for each journey
    def make_signature(group):
        # A journey is represented as a tuple of pairs
        return tuple(zip(group[index_col], group[next_col]))

    signatures = df.groupby("journey_id_real").apply(make_signature)

    # 3. Keep only unique signatures (drop duplicate journeys)
    unique_journey_ids = signatures.drop_duplicates().index

    # 4. Filter original dataframe to only those journeys
    result = df[df["journey_id_real"].isin(unique_journey_ids)].copy()

    # Cleanup optional:
    result.drop(columns=["new_journey"], inplace=True)

    return result


result = dedupe_variable_length_journeys(face_paths)

testdf = raw_paths[raw_paths['path_id']=='5852_3541'].copy()

y[testdf['index'].values]

indices = list(testdf['index'].values)
final_ind = list([testdf['next_index'].values[-1]])

plot_paths(np.array(X),list(y),indices+final_ind)

for paths in path_dict.values():
    print(len(set(paths['path_id'])))


from tqdm import tqdm
import math

from random import sample, seed

random_seed = 42
seed(random_seed)

path_dict = {
    'raw': raw_paths,
    'pca': pca_paths,
    'face': result,
}

data_dict ={}

common_path_ids = set.intersection(*[
    set([p_id.split("_")[0] for p_id in paths['path_id']])
    for paths in path_dict.values()
])

x_sample = sample(list(common_path_ids),n_train)

sample_path_ids = {}



for path_name,paths in path_dict.items():

    sample_path_ids[path_name] = [sample([p for p in paths['path_id'].unique() if p.split("_")[0] == x],1)[0] for x in x_sample]

    data_dict[path_name] = {
    'X':[],
    'Y':[],
    'K':{
        'X':[],
        'Y':[],
        'K':[],
        }
    }   

    print("Path Type: ",path_name)
    both_ways = False
    # path_ids = [ p for p in paths['path_id'].unique() if p.split("_")[0] in x_sample]
    

    for path_id in tqdm(sample_path_ids[path_name]): #sample(list(path_ids),int(math.floor(n_train/2)))):
        X_0,X_1 = [int(id) for id in path_id.split("_")]
        Y_0,Y_1 = y[int(X_0)],y[int(X_1)]
        
        paths_subset = paths[paths['path_id']==path_id].copy()
        
        full_path = list(paths_subset['index'].values) + [paths_subset['next_index'].values[-1]]

        # print("Full Path & Label:")
        # for pair in list(zip(full_path,[y[index] for index in full_path])):
        #     print(f"    {pair}")
        
        change_indice = np.where(y[full_path] == Y_1)[0][0]
        # print('indices: ',full_path,'\nChange: ',change_indice)

        Z0 = {
            'X':full_path[:change_indice],
            # 'Y':y_df.loc[paths_subset['index']]['y'],
            'Y':y[full_path][:change_indice],
            'K':full_path[1:change_indice+1],
        }
        Z1 = {
            'X':full_path[change_indice:][::-1],
            # 'Y':y_df.loc[paths_subset['index']]['y'],
            'Y':list(y[full_path])[change_indice:][::-1],
            'K':full_path[change_indice-1:-1][::-1],
        }

        # path_data = {
        #     'X':[X_0,X_1],
        #     'Y':[Y_0,Y_1],
        #     'Z':{
        #         'X':[Z0['X'],Z1['X']],
        #         'Y':[Z0['Y'],Z1['Y']],
        #         'K':[Z0['K'],Z1['K']],
        #     }
        # }

        if both_ways:
            data_dict[path_name]['X'].extend([X_0, X_1])
            data_dict[path_name]['Y'].extend([Y_0, Y_1])

            data_dict[path_name]['K']['X'].append(Z0['X'])
            data_dict[path_name]['K']['X'].append(Z1['X'])

            data_dict[path_name]['K']['Y'].append(Z0['Y'])
            data_dict[path_name]['K']['Y'].append(Z1['Y'])

            data_dict[path_name]['K']['K'].append(Z0['K'])
            data_dict[path_name]['K']['K'].append(Z1['K'])
        else:
            data_dict[path_name]['X'].append(X_0)
            data_dict[path_name]['Y'].append(Y_0)

            data_dict[path_name]['K']['X'].append(Z0['X'])
            data_dict[path_name]['K']['Y'].append(Z0['Y'])
            data_dict[path_name]['K']['K'].append(Z0['K'])


        
        

assert data_dict['raw']['X'] == data_dict['pca']['X']
assert data_dict['raw']['Y'] == data_dict['pca']['Y']
assert data_dict['raw']['X'] == data_dict['face']['X']


# from tqdm import tqdm
# import math

# from random import sample, seed

# random_seed = 42
# seed(random_seed)

# path_dict = {
#     'raw': raw_paths,
#     'pca': pca_paths,
#     'face': result,
# }

# data_dict ={}

# for path_name,paths in path_dict.items():

#     data_dict[path_name] = {
#     'X':[],
#     'Y':[],
#     'K':{
#         'X':[],
#         'Y':[],
#         'K':[],
#         }
#     }   

#     print("Path Type: ",path_name)
#     path_ids = paths['path_id'].unique()

#     for path_id in tqdm(sample(list(path_ids),int(math.floor(n_train/2)))):
#         X_0,X_1 = [int(id) for id in path_id.split("_")]
#         Y_0,Y_1 = y[int(X_0)],y[int(X_1)]

#         paths_subset = paths[paths['path_id']==path_id].copy()
#         full_path = list(paths_subset['index'].values) + [paths_subset['next_index'].values[-1]]

#         # print("Full Path & Label:")
#         # for pair in list(zip(full_path,[y[index] for index in full_path])):
#         #     print(f"    {pair}")
        
#         change_indice = np.where(y[full_path] == Y_1)[0][0]
#         # print('indices: ',full_path,'\nChange: ',change_indice)

#         Z0 = {
#             'X':full_path[:change_indice],
#             # 'Y':y_df.loc[paths_subset['index']]['y'],
#             'Y':y[full_path][:change_indice],
#             'K':full_path[1:change_indice+1],
#         }
#         Z1 = {
#             'X':full_path[change_indice:][::-1],
#             # 'Y':y_df.loc[paths_subset['index']]['y'],
#             'Y':list(y[full_path])[change_indice:][::-1],
#             'K':full_path[change_indice-1:-1][::-1],
#         }

#         path_data = {
#             'X':[X_0,X_1],
#             'Y':[Y_0,Y_1],
#             'Z':{
#                 'X':[Z0['X'],Z1['X']],
#                 'Y':[Z0['Y'],Z1['Y']],
#                 'K':[Z0['K'],Z1['K']],
#             }
#         }
        
#         data_dict[path_name]['X'].extend([X_0, X_1])
#         data_dict[path_name]['Y'].extend([Y_0, Y_1])

#         data_dict[path_name]['K']['X'].append(Z0['X'])
#         data_dict[path_name]['K']['X'].append(Z1['X'])

#         data_dict[path_name]['K']['Y'].append(Z0['Y'])
#         data_dict[path_name]['K']['Y'].append(Z1['Y'])

#         data_dict[path_name]['K']['K'].append(Z0['K'])
#         data_dict[path_name]['K']['K'].append(Z1['K'])



        
        

import pickle as pkl

with open(f'data/paths/mnist_paths_data_split_{n_train}paths.pkl', 'wb') as f:
    pkl.dump(data_dict, f)  

n_train

raw_paths

paths = {
    'raw':{'original':{'inputs':raw_paths}},
    'pca':{'original':{'inputs':pca_paths}},
    'face':{'original':{'inputs':face_paths}}
}

for path_type, path in paths.items():
    paths[path_type]['split'] = {'outputs': data_dict[path_type]}
    
    print(f"{path_type} paths, Split: {len(paths[path_type]['split']['outputs'])}")


def compile_mnist(X,y,cf_X,cf_y):

    magnitude = np.ones(len(cf_X))
    magnitude = np.expand_dims(magnitude, axis=1)
    
    print(f'Returning {len(X)} samples with {len(cf_y)} counterfactuals')
    
    return {'X':np.array(X),
            'Y':list(y),
            'K':{
                'vector':cf_X,
                 'label':cf_y,
                 'magnitude':np.expand_dims(magnitude,axis=1), 
                 }}

COMMON = True

for path_type, path in data_dict.items():
    print(f"Compiling {path_type} data split...")
    
    dataset = paths[path_type]['split']['outputs']

    if boundary_only:
        for key,val in dataset['K'].items():
            dataset['K'][key] = [[v[-1]] for v in val]
            # print(key,val)
    
        
    X_train_sample = np.array(X.iloc[dataset['X']])
    y_train_sample = np.array(dataset['Y']).squeeze()


    # Stripping html from unprocessed text, just to clean it up
    print('\ntrain_Data')
    cf_train={'original': {
                'X':np.array(X_train_sample),
                'Y':list(y_train_sample),
                'K':{
                    'X':[np.array(X.iloc[i]) for i in dataset['K']['X']],
                    'Y':[y[i] for i in dataset['K']['Y']],
                    'K':[np.array(X.iloc[i]) for i in dataset['K']['K']],
                    }}}

    print('\ndev_Data')
    cf_dev={'original': {
                'X':np.array(X_dev),
                'Y':list(y_dev),
                'K':{
                    'X':np.array(np.zeros_like(X_dev)),
                    'Y':np.array(np.zeros_like(y_dev)),
                    'K':np.array(np.zeros_like(X_dev)),
                    }}}
    
            
    print('\ntest_Data')
    cf_test={'original': {
                'X':np.array(X_test),
                'Y':list(y_test),
                'K':{
                    'X':np.array(np.zeros_like(X_test)),
                    'Y':np.array(np.zeros_like(y_test)),
                    'K':np.array(np.zeros_like(X_test)),
                    }}}
    
    print('\ncontrol_Data')
    flattened_X = [x for xs in dataset['K']['X'] for x in xs]
    flattened_Y = [y for ys in dataset['K']['Y'] for y in ys]


    cf_control={'original': {
                'X':np.array(flattened_X),
                'Y':list(flattened_Y),
                'K':{
                    'X':np.array(np.zeros_like(flattened_X)),
                    'Y':np.array(np.zeros_like(flattened_Y)),
                    'K':np.array(np.zeros_like(flattened_X)),
                    }}}
    

    """
    ########################################################################################################################
    Save embeddings
    ########################################################################################################################
    """

    pickle_data = {'train':cf_train,'test':cf_test,'dev':cf_dev, 'control':cf_control,'n_classes': 10}
    
    embedding_path = f"data/integer_len{len(dataset['X'])}__connected_{str(connected_paths)}__path_{path_type}__split__boundary_{str(boundary_only)}.pkl"
    print(f"Saving to {embedding_path}")
    with open(embedding_path, 'wb') as file:
        pkl.dump(pickle_data, file)

    if visualise:
        print(f"PATH TYPE: {path_type} | PATH LENGTH: {path_length}")
        sample_size = 5
        # for i,K in enumerate(cf_train['original']['K']['X'][0:sample_size]):           
        for i in range(5):
            print(f'\n\nRow {i}')
            plot_paths(np.array(X),list(y),dataset['K']['X'][i])
            plot_paths(np.array(X),list(y),dataset['K']['K'][i]) 
            # plot_paths([cf_train['original']['X'][i]],[cf_train['original']['Y'][i]],[0])
            # plot_paths(K,cf_train['original']['K']['Y'][i],np.arange(len(K)))



[len(d) for d in dataset['K']['X']]

for i in range(5):
    print(f'\n\nRow {i}')
    plot_paths(np.array(X),list(y),dataset['K']['X'][i])
    plot_paths(np.array(X),list(y),dataset['K']['K'][i])

