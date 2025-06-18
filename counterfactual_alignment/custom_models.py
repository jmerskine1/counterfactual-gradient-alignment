#DEFINE Neural network architecture
# Input layer | Dense Linear | Tanh Activation | Dense Linear Output

import flax
from flax import linen as nn
import jax.numpy as np
from jax import random
from flax.core.frozen_dict import unfreeze
from jax.nn.initializers import glorot_normal, normal

class MLP(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(features=self.num_hidden)
        # self.relu = nn.relu()
        self.fc2 = nn.Dense(features=self.num_outputs)
        # self.sigmoid = nn.sigmoid()  # For binary classification

    def __call__(self, x,train=False):
        
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.fc2(x)
        # x = nn.sigmoid(x)
        return x.squeeze(axis=-1)


class SimpleClassifier(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs) 
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x,train=False):
        x = self.linear1(x)      
        x = nn.tanh(x)
        x = self.dropout(x, deterministic= not train)
        x = self.linear2(x)
  
        return x.squeeze(axis=-1)


class CNN(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3,), padding="SAME", name="CONV1")
        self.conv2 = nn.Conv(features=16, kernel_size=(3,), padding="SAME", name="CONV2")
        self.dense1 = nn.Dense(features=1)
        self.linear1 = nn.Dense(1, name="DENSE")

    def __call__(self, inputs, train=False):
        # print('x',np.shape(inputs))
        x = nn.relu(self.conv1(inputs))
        # print('x',np.shape(x))
        # x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        
        # print('x',np.shape(x))
        x = x.reshape((x.shape[0], -1))
        # print('x',np.shape(x))
        x = self.linear1(x)
        # print('x',np.shape(nn.softmax(x)))
        # print(x)
        # print(nn.softmax(x))
        # Output layer with sigmoid activation
        predictions = self.dense1(x)
        predictions = nn.sigmoid(predictions)

        return predictions
        # x = nn.relu(x)
        # return x


class GSPaperNew(nn.Module):
    num_hidden: int   # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    def setup(self):
        # Define the layers for the network
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)
        self.linear3 = nn.Dense(features=64)
        self.linear4 = nn.Dense(features=128)

        # Define Conv1D layers
        self.conv1 = nn.Conv(features=128, kernel_size=(7,), padding='VALID', strides=(3,))
        self.conv2 = nn.Conv(features=64, kernel_size=(7,), padding='SAME', strides=(1,),
                             kernel_init=glorot_normal(), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(7,), padding='SAME',
                             kernel_init=glorot_normal(), use_bias=True)

        # Dropout layers
        self.dropout = nn.Dropout(rate=0.5)
        self.dropout2 = nn.Dropout(rate=0.5)

    def __call__(self, x, train=False):
        # Ensure input is 3D for Conv1D: (batch_size, sequence_length, channels)
        # print('1',np.shape(x))
        
        # if x.ndim == 2:  # If input is (batch_size, sequence_length)
        #     x = np.expand_dims(x, axis=-1)  # Add a channel dimension: (batch_size, sequence_length, 1)

        
        # # First dense layer
        # x = self.linear4(x)
        # x = self.dropout(x, deterministic=not train)
        # # print('1',x)
        # # Conv1D layer
        # # x = self.conv3(x)
        # print('afterconv',np.shape(x))
        # x = nn.relu(x)
        
        # Global Max Pooling across the sequence length dimension
        # x = np.mean(x, axis=1)

        # print('aftermax',np.shape(x))

        # Dense hidden layer
        x = self.linear3(x)
        x = nn.relu(x)
        # x = self.dropout2(x, deterministic=not train)

        # Output layer
        x = self.linear2(x)
        
        return nn.sigmoid(x).squeeze(axis=-1)
    


class GSPaper2(nn.Module):
    num_hidden: int   # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    def setup(self):
        # Define the layers for the network
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)
        self.linear3 = nn.Dense(features=256)
        self.linear4 = nn.Dense(features=128)

        # Define Conv1D layers
        self.conv1 = nn.Conv(features=128, kernel_size=(7,), padding='VALID', strides=(3,))
        self.conv2 = nn.Conv(features=64, kernel_size=(7,), padding='SAME', strides=(1,),
                             kernel_init=glorot_normal(), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(7,), padding='SAME',
                             kernel_init=glorot_normal(), use_bias=True)

        # Dropout layers
        self.dropout = nn.Dropout(rate=0.3)
        self.dropout2 = nn.Dropout(rate=0.5)

    def __call__(self, x, train=False):
        # Ensure input is 3D for Conv1D: (batch_size, sequence_length, channels)
        # print('1',np.shape(x))
        
        # if x.ndim == 2:  # If input is (batch_size, sequence_length)
        #     x = np.expand_dims(x, axis=-1)  # Add a channel dimension: (batch_size, sequence_length, 1)

        
        # # First dense layer
        # x = self.linear4(x)
        # x = self.dropout(x, deterministic=not train)
        # # print('1',x)
        # # Conv1D layer
        # # x = self.conv3(x)
        # print('afterconv',np.shape(x))
        # x = nn.relu(x)
        
        # Global Max Pooling across the sequence length dimension
        # x = np.mean(x, axis=1)

        # print('aftermax',np.shape(x))

        # Dense hidden layer
        x = self.linear3(x)
        x = nn.relu(x)
        x = self.dropout2(x, deterministic=not train)

        x = self.linear4(x)
        x = nn.relu(x)

        x = self.dropout(x, deterministic=not train)

        # Output layer
        x = self.linear2(x)
        
        return nn.sigmoid(x).squeeze(axis=-1)
    

class GSPaper3(nn.Module):
    num_hidden: int   # Number of hidden neurons
    num_outputs: int  # Number of output neurons

    def setup(self):
        # Define the layers for the network
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)
        self.linear3 = nn.Dense(features=128)
        self.linear4 = nn.Dense(features=64)

        # Define Conv1D layers
        self.conv1 = nn.Conv(features=128, kernel_size=(7,), padding='VALID', strides=(3,))
        self.conv2 = nn.Conv(features=64, kernel_size=(7,), padding='SAME', strides=(1,),
                             kernel_init=glorot_normal(), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(7,), padding='SAME',
                             kernel_init=glorot_normal(), use_bias=True)

        # Dropout layers
        self.dropout = nn.Dropout(rate=0.5)
        self.dropout2 = nn.Dropout(rate=0.5)

    def __call__(self, x, train=False):
        
        # Dense hidden layer
        x = self.linear3(x)
        x = nn.relu(x)
        x = self.dropout2(x, deterministic=not train)

        x = self.linear4(x)
        x = nn.relu(x)

        x = self.dropout(x, deterministic=not train)

        # Output layer
        x = self.linear2(x)
        
        return nn.sigmoid(x).squeeze(axis=-1)
        


        

class GSPaper4(nn.Module):
    # num_outputs: int  # Number of output neurons
    vocabulary_size: int
    embedding_size: int

    @nn.compact
    def __call__(self, x, train=False):
        # Embedding layer
        x = nn.Embed(self.vocabulary_size,self.embedding_size,embedding_init=glorot_normal())(x)
        
        
        # x = nn.relu(x)
        
        # print("Embed: ",np.shape(x))
        x = np.average(x, axis=1)
        
        
        # Define a single dense layer
        # Apply sigmoid activation
        x = nn.Dense(features=1)(x)
        # print('pre_sigmoid',x)
        # x = nn.sigmoid(x)
        output = nn.sigmoid(x)
        # print('post_sigmoid',x)
        

        return output


# Define the model
class BagOfWordsClassifier(nn.Module):
    vocabulary_size: int = 20001  # Limit vocabulary to 20,000 most frequent words
    embedding_size: int = 50      # Embedding size of 50

    def setup(self):
        # Define the layers for the network
        self.linear1 = nn.Dense(features=1,kernel_init=nn.initializers.glorot_normal())  # Shape: (batch_size, 1)
        
        self.embed = nn.Embed(
            num_embeddings=self.vocabulary_size,  # +2 for padding and unknown tokens
            features=self.embedding_size,
            embedding_init=nn.initializers.glorot_normal())
        
        self.dropout = nn.Dropout(rate=0.5)
        # self.batch_norm1 = nn.BatchNorm(momentum=0.9, epsilon=1e-5)

    def __call__(self, x, train=False):
        # x: (batch_size, max_sequence_length), where each element is a word index

        # Create a mask where valid tokens are 1 and padding tokens (e.g., 0) are 0
        mask = (x != -1).astype(np.float32)  # Assuming -1 is used for padding

        # Embedding layer
        embeddings = self.embed(x)  # Shape: (batch_size, seq_length, embedding_size)
        
        # Apply mask to embeddings
        embeddings = embeddings * mask[..., None]  # Shape: (batch_size, seq_length, embedding_size)

        # Compute the sum of embeddings over valid tokens
        embedding_sum = np.sum(embeddings, axis=1)  # Shape: (batch_size, embedding_size)

        # Compute the number of valid tokens in each sequence
        valid_tokens_count = np.sum(mask, axis=1, keepdims=True)  # Shape: (batch_size, 1)
        valid_tokens_count = np.maximum(valid_tokens_count, 1)  # Avoid division by zero

        # Compute the average embedding
        x = embedding_sum / valid_tokens_count  # Shape: (batch_size, embedding_size)
        x = self.dropout(x, deterministic=not train)
        
        # x = self.linear2(x)
        # x = self.dropout(x, deterministic=not train)
        # Linear classifier
        logits = self.linear1(x)
        
        # Sigmoid output
        # probs = nn.sigmoid(logits).squeeze(axis=-1)  # Shape: (batch_size, 1), values between 0 and 1

        return logits.squeeze(axis=-1)
        # return probs


class BagOfWordsClassifierSimple(nn.Module):
    vocabulary_size: int = 20001
    embedding_size: int = 50

    def setup(self):
        self.linear1 = nn.Dense(features=1, kernel_init=nn.initializers.glorot_normal())
        self.embed = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size,
                              embedding_init=nn.initializers.glorot_normal())
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, train=False):
        # x: (batch_size, seq_length), assumed padded with 0
        mask = (x != 0).astype(np.float32)  # Padding mask
        
        embeddings = self.embed(x)  # (batch_size, seq_length, embedding_size)
        
        embeddings = embeddings * mask[..., None]

        summed = np.sum(embeddings, axis=1)
        lengths = np.sum(mask, axis=1, keepdims=True)
        x = summed / np.maximum(lengths, 1)

        x = self.dropout(x, deterministic=not train)
        logits = self.linear1(x)
        return logits.squeeze(axis=-1)


# Define the model
class BagOfWordsClassifierSingle(nn.Module):
    vocabulary_size: int = 20001  # Limit vocabulary to 20,000 most frequent words
    embedding_size: int = 50      # Embedding size of 50

    def setup(self):
        # Define the layers for the network
        self.linear1 = nn.Dense(features=1,kernel_init=nn.initializers.glorot_normal())  # Shape: (batch_size, 1)
        
        self.embed = nn.Embed(
            num_embeddings=self.vocabulary_size,  # +2 for padding and unknown tokens
            features=self.embedding_size,
            embedding_init=nn.initializers.glorot_normal())
        
        self.dropout = nn.Dropout(rate=0.5)
        # self.batch_norm1 = nn.BatchNorm(momentum=0.9, epsilon=1e-5)

    def __call__(self, x, train=False):
        # x: (batch_size, max_sequence_length), where each element is a word index
        
        # Create a mask where valid tokens are 1 and padding tokens (e.g., 0) are 0
        mask = (x != -1).astype(np.float32)  # Assuming 0 is used for padding
        
        # Embedding layer
        embeddings = self.embed(x)  # Shape: (batch_size, seq_length, embedding_size)
        
        # Apply mask to embeddings
        embeddings = embeddings.T * mask  # Shape: (batch_size, seq_length, embedding_size)
        
        # Compute the sum of embeddings over valid tokens
        embedding_sum = np.sum(embeddings,axis=1)  # Shape: (batch_size, embedding_size)
        
        # Compute the number of valid tokens in each sequence
        valid_tokens_count = np.sum(mask)  # Shape: (batch_size, 1)
        valid_tokens_count = np.maximum(valid_tokens_count, 1)  # Avoid division by zero
        
        # Compute the average embedding
        x = embedding_sum / valid_tokens_count  # Shape: (batch_size, embedding_size)
        
        # x = self.dropout(x, deterministic=not train)
        
        # x = self.linear2(x)
        # x = self.dropout(x, deterministic=not train)
        # Linear classifier
        logits = self.linear1(x)
        
        # print("mask",np.shape(mask))
        # print("embed",np.shape(embeddings))
        # print("embed_masked",np.shape(embeddings))
        # print("embed_sum",np.shape(embedding_sum))
        # print('valid',valid_tokens_count)
        # print('x',np.shape(x))
        # print('logits',np.shape(logits))
        
        # Sigmoid output
        # probs = nn.sigmoid(logits).squeeze(axis=-1)  # Shape: (batch_size, 1), values between 0 and 1
        # print(np.sum(probs))
        return logits
    
class GSPaper5(nn.Module):
    # num_outputs: int  # Number of output neurons
    vocabulary_size: int
    embedding_size: int

    @nn.compact
    def __call__(self, x, train=False):
        # Embedding layer
        # print("IN: ",np.shape(x))

        # Create a mask where -1 (padded) values are 0, and valid tokens are 1
        mask = (x != -1).astype(np.float32)
        

        x = nn.Embed(self.vocabulary_size,self.embedding_size)(x)
        # print("POstembed: ",np.shape(x))
        
        # Apply the mask to the embeddings (set embeddings for -1 tokens to 0)
        x = x * mask[..., None]  # Broadcasting mask to (batch_size, seq_length, 1)
        # print(mask)
        # print("aftermask: ",np.shape(x))
        # Compute the sum of embeddings for valid tokens
        embedding_sum = np.sum(x, axis=1)  # Shape (batch_size, embedding_size)
        # print('sum shape',np.shape(embedding_sum))
        # Compute the number of valid tokens for each sequence
        valid_tokens_count = np.sum(mask, axis=1, keepdims=True)  # Shape (batch_size, 1)
        # print("Validt",np.shape(valid_tokens_count))
        # Prevent division by zero by ensuring valid_tokens_count is at least 1
        valid_tokens_count = np.maximum(valid_tokens_count, 1)
        # print(valid_tokens_count)
        # Compute the average embedding over valid tokens
        x = embedding_sum / valid_tokens_count  # Shape (batch_size, embedding_size)
        # print("aftersum and average: ",np.shape(x))
        # print("Embed: ",np.shape(x))
        # x = np.mean(x, axis=1)
        
        # Define a single dense layer
        x = nn.Dense(features=1,kernel_init=glorot_normal())(x)  # Linear layer (Wx + b)
        # print('pre_sigmoid',x)
        # Apply sigmoid activation
        x = nn.sigmoid(x)
        # print("out: ", np.shape(x))
        

        return x

class GSPaper(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons


    # # # Conv1D + global max pooling
    # # x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    # x = layers.Conv1D(64, 7, padding="valid", activation="relu", strides=3)(x)
    # x = layers.GlobalMaxPooling1D()(x)

    # # We add a vanilla hidden layer:
    # x = layers.Dense(64, activation="relu")(x)
    # x = layers.Dropout(0.9)(x)
    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden)
        self.linear2 = nn.Dense(features=self.num_outputs)
        self.linear3 = nn.Dense(features=16)
        self.linear4 = nn.Dense(features=128)
        self.conv1 = nn.Conv(128,[7])
        self.conv2 = nn.Conv(features=64, kernel_size=(7,), padding='SAME', strides=(1,),
                    kernel_init=glorot_normal(), use_bias=True)
        self.conv3 = nn.Conv(features=64, kernel_size=(7,), padding='SAME',
                    kernel_init=glorot_normal(), use_bias=True)
        self.dropout = nn.Dropout(rate=0.5)
        self.dropout2 = nn.Dropout(rate=0.9)

    def __call__(self, x,train=False):
        x = self.linear4(x)
        x = self.dropout(x , deterministic=not train)
        
        # Conv1D layer
        x = self.conv3(x)

        x = nn.relu(x)
        # Global Max Pooling (done by reducing max over the time dimension)
        # x = np.max(x, axis=1)
        
        # Dense hidden layer
        x = self.linear3(x)
        
        x = nn.relu(x)
        x = self.dropout2(x, deterministic=not train)

        # Output layer
        x = self.linear2(x)

        return nn.sigmoid(x)

class TextClassifier(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Assuming x is the input tensor of shape [batch_size, 50]
        # where each input x is a vector of length 50.

        # First convolutional layer
        x = nn.Conv(features=128, kernel_size=(7,), padding="VALID", strides=(3,))(x)
        x = np.max(x, axis=1)

        # Output layer with sigmoid activation
        predictions = nn.Dense(features=128)(x)
        predictions = nn.sigmoid(predictions)

        return predictions
    

class TextClassifierHard(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Assuming x is the input tensor of shape [batch_size, 50]
        # where each input x is a vector of length 50.

        # Apply dropout to the input layer
        x = nn.Dropout(rate=0.5)(x, deterministic=False)

        # First convolutional layer
        x = nn.Conv(features=128, kernel_size=(7,), padding="VALID", strides=(3,))(x)
        x = nn.relu(x)

        # Second convolutional layer
        x = nn.Conv(features=128, kernel_size=(7,), padding="VALID", strides=(3,))(x)
        x = nn.relu(x)

        # Global Max Pooling
        x = np.max(x, axis=1)

        # Dense layer with ReLU activation
        x = nn.Dense(features=128)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=False)

        # Output layer with sigmoid activation
        predictions = nn.Dense(features=128)(x)
        predictions = nn.sigmoid(predictions)

        return predictions

class TextClassifierEmbeddings(nn.Module):
# https://www.machinelearningnuggets.com/handling-state-in-jax-flax-batchnorm-and-dropout-layers/#evaluation-step
# worked example of batchnorm
    @nn.compact
    def __call__(self, x,train=False):
        # Embedding layer
        
        x = nn.Embed(
            num_embeddings=20000, # Number of unique tokens
            features=50,  # Dimension of the embedding vectors
            embedding_init= glorot_normal(), #nn.initializers.uniform(scale=1.0)
            
        )(x)
        x = np.mean(x, axis=1) 
        x = nn.relu(x)
        x = nn.Dropout(0.7, deterministic=not train)(x)

        # x = nn.BatchNorm(use_running_average=not train)(x)
        
        
        # print("after embedding:", x.shape)
        # x = nn.Conv(features=16, kernel_size=(5,), padding="VALID", strides=(1,))(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        
        
        x = nn.Dense(features=2)(x)  # Consolidate features
        
        x = nn.leaky_relu(x)
        
        # Apply dropout to the embedded layer
        
        x = nn.Dropout(0.8, deterministic=not train)(x)

        x = nn.Dense(features=1)(x)  # Consolidate features
        
        return x.squeeze(axis=-1)



class TextClassifierEmbeddingsSetfit(nn.Module):
# https://www.machinelearningnuggets.com/handling-state-in-jax-flax-batchnorm-and-dropout-layers/#evaluation-step
# worked example of batchnorm
    @nn.compact
    def __call__(self, x,train=False):
        # print("SHAPE of x",np.shape(x))
        # Embedding layer
        # x = nn.Embed(
        #     num_embeddings=20000, # Number of unique tokens
        #     features=50,  # Dimension of the embedding vectors
        #     embedding_init= glorot_normal(), #nn.initializers.uniform(scale=1.0)
        # )(x)
        x = nn.Conv(features=128, kernel_size=(7,), padding="SAME")(x) #strides=(3,))(x)
        # print("SHAPE of x postconv",np.shape(x))
        x = np.mean(x, axis=1) 
        x = nn.relu(x)
        x = nn.Dropout(0.7, deterministic=not train)(x)

        # x = nn.BatchNorm(use_running_average=not train)(x)
        
        
        # print("after embedding:", x.shape)
        # x = nn.Conv(features=16, kernel_size=(5,), padding="VALID", strides=(1,))(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        print(0,np.shape(x))
        x = nn.leaky_relu(x)
        
        print(1,np.shape(x))
        x = nn.Dense(features=2)(x)  # Consolidate features
        
        x = nn.leaky_relu(x)
        
        # Apply dropout to the embedded layer
        
        x = nn.Dropout(0.8, deterministic=not train)(x)

        x = nn.Dense(features=1)(x)  # Consolidate features
        
        return x.squeeze(axis=-1)

class TextClassifierEmbeddingsBatch(nn.Module):
# https://www.machinelearningnuggets.com/handling-state-in-jax-flax-batchnorm-and-dropout-layers/#evaluation-step
# worked example of batchnorm
    @nn.compact
    def __call__(self, x,train=False):
        # Embedding layer
        x = nn.Embed(
            num_embeddings=20000, # Number of unique tokens
            features=50,  # Dimension of the embedding vectors
            embedding_init= glorot_normal(), #nn.initializers.uniform(scale=1.0)
        )(x)
        x = np.mean(x, axis=1) 

        # x = nn.BatchNorm(use_running_average=not train)(x)
        
        
        # print("after embedding:", x.shape)
        # x = nn.Conv(features=16, kernel_size=(5,), padding="VALID", strides=(1,))(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        
        
        x = nn.Dense(features=100)(x)  # Consolidate features
        
        x = nn.leaky_relu(x)
        
        # Apply dropout to the embedded layer
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.Dropout(0.5, deterministic=not train)(x)

        x = nn.Dense(features=1)(x)  # Consolidate features
        
        return x.squeeze(axis=-1)

class TheOverfitter(nn.Module):

    @nn.compact
    def __call__(self, x,train=False):
        # Embedding layer
        x = nn.Embed(
            num_embeddings=20000, # Number of unique tokens
            features=50,  # Dimension of the embedding vectors
            embedding_init= glorot_normal() #nn.initializers.uniform(scale=1.0)
        )(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        
        # print("after embedding:", x.shape)
        x = nn.Conv(features=32, kernel_size=(5,), padding="VALID", strides=(1,))(x)
        # x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.leaky_relu(x)
        

        x = nn.Dropout(rate=0.8)(x, deterministic=not train)
        
        # print("after conv1:", x.shape)
        # x = nn.Conv(features=12, kernel_size=(3,), padding="SAME", strides=(1,))(x)
        # x = nn.relu(x)
        # print("after conv2:", x.shape)
        x = np.mean(x, axis=1) 
        x = nn.leaky_relu(x)
        # print("after mean:", x.shape)
        
        # x = nn.max_pool(inputs=x,window_shape=(28,),strides=(1,))
        # print("after max pool:", x.shape)
        
        x = nn.Dense(features=8)(x)  # Consolidate features
        
        x = nn.leaky_relu(x)
        # x = np.mean(x, axis=1) 
        
        # Apply dropout to the embedded layer
        x = nn.Dropout(rate=0.8)(x, deterministic=not train)

        x = nn.Dense(features=1)(x)  # Consolidate features
        
        return x.squeeze(axis=-1)


class TextClassifierEmbeddingsOld(nn.Module):

    @nn.compact
    def __call__(self, x,train=False):
        # Embedding layer
        x = nn.Embed(
            num_embeddings=20000, # Number of unique tokens
            features=50,  # Dimension of the embedding vectors
            embedding_init=nn.initializers.uniform(scale=1.0)
        )(x)
        x = nn.Conv(features=128, kernel_size=(5,), padding="SAME", strides=(1,))(x)
        x = nn.Conv(features=64, kernel_size=(3,), padding="SAME", strides=(1,))(x)
        # x = np.mean(x, axis=1) 
        
        # Apply dropout to the embedded layer
        x = nn.Dropout(rate=0.5)(x, deterministic=not train)
        x = x[:, None, :]
        
        # First convolutional layer
        
        # x = nn.relu(x)
        
        # Second convolutional layer
        # x = nn.Conv(features=16, kernel_size=(3,), padding="SAME", strides=(1,))(x)
        # x = nn.relu(x)

        # Global Max Pooling
        x = np.max(x, axis=2)
        
        # Dense layer with ReLU activation
        # x = nn.Dense(features=128)(x)

        # x = nn.relu(x)

        x = nn.Dropout(rate=0.5)(x, deterministic=not train)

        x = nn.Dense(features=1)(x)  # Consolidate features
        # x = nn.relu(x)
        # predictions = nn.Dense(features=128)(x)  # Assuming binary classification
        # print("AFTER DENSE: ",predictions.shape)

        
        # predictions = nn.sigmoid(x)
        # Output layer with sigmoid activation
        # predictions = nn.Dense(features=128)(x)
        
        # predictions = nn.sigmoid(predictions)
        

        return x

class SimpleGPTModel(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Dense(features=1)(x)
        return x

# Check if a simple model can learn anything
class SimpleClassifierNew(nn.Module):

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=66)
        self.linear2 = nn.Dense(features=1) 
        

    def __call__(self, x, train=False):
        # Perform the calculation of the model to determine the prediction
        x = self.linear1(x)
        x = nn.tanh(x)
        x = self.linear2(x)
        return x

custom_models = {'simple':SimpleClassifier, 'gradient_supervision':GSPaper, 'GPTattempt':TextClassifierHard}