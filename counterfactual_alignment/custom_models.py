#DEFINE Neural network architecture
# Input layer | Dense Linear | Tanh Activation | Dense Linear Output

import flax
from flax import linen as nn
import jax.numpy as np

import jax
from jax import debug
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
  
        return x, None 
    

class MultiClassClassifier(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden,kernel_init=glorot_normal())
        self.linear2 = nn.Dense(features=self.num_outputs,kernel_init=glorot_normal()) 
        self.dropout = nn.Dropout(rate=0.1)

    def __call__(self, x,train=False):
        x = self.linear1(x)      
        x = nn.relu(x)
        # x = self.dropout(x, deterministic= not train)
        x = self.linear2(x)
  
  
        return x, None


class SimpleClassifier_v2(nn.Module):
    num_hidden : int   # Number of hidden neurons
    num_outputs : int  # Number of output neurons

    def setup(self):
        # Create the modules we need to build the network
        # nn.Dense is a linear layer
        self.linear1 = nn.Dense(features=self.num_hidden,kernel_init=glorot_normal())
        self.linear2 = nn.Dense(features=self.num_outputs,kernel_init=glorot_normal()) 
        self.dropout = nn.Dropout(rate=0.1)

    def __call__(self, x,train=False):
        x = self.linear1(x)      
        x = nn.relu(x)
        # x = self.dropout(x, deterministic= not train)
        x = self.linear2(x)
        
  
        return x, None



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
    vocabulary_size: int = 20002
    embedding_size: int = 50

    
    def setup(self):
        self.linear1 = nn.Dense(features=1, kernel_init=nn.initializers.glorot_normal())#,bias_init=nn.initializers.normal(stddev=0.1))
        self.embed = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size,
                              embedding_init=nn.initializers.glorot_normal())
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, train=False):
        
        x = np.asarray(x)  # Ensure x is a numpy array
        # debug.print('x: shape: {sx},{maxx},{minx}',sx=np.shape(x),maxx = np.max(x),minx = np.min(x))
        # x: (batch_size, seq_length), assumed padded with 0
        mask = (x != 0).astype(np.float32)  # Padding mask

        # debug.print('mask: {x}',x=np.mean(mask))
        embeddings = self.embed(x)  # (batch_size, seq_length, embedding_size)
        
        # debug.print('embeddings: {x}',x=np.mean(embeddings))
        embeddings = embeddings * mask[..., None]
        # jax.debug.print("embeds: {}",np.shape(embeddings))
        # debug.print('embeddings2: {x}',x=np.mean(embeddings))
        summed = np.sum(embeddings, axis=1)

        lengths = np.sum(mask, axis=1, keepdims=True)
        # jax.debug.print("lengths: {}",np.shape(lengths))
        # debug.print('shape of input: {y}\n lengths: {x}',x=np.shape(lengths),y = np.shape(x))
        x = summed / np.maximum(lengths, 1) 
        # jax.debug.print("x: {}",x)
        x_drop = self.dropout(x, deterministic=not train)
        logits = self.linear1(x_drop)
        # jax.debug.print('logits:{x}',x=logits)
        # return logits.squeeze(axis=-1), x
        
        return logits, x


# class MulticlassEmbeddingOnlyModel(nn.Module):
#     num_classes: int
    
#     @nn.compact
#     def __call__(self, embedded_inputs,train=False):  # embedded_inputs: (batch, embed_dim)
#         logits = nn.Dense(features=self.num_classes, name='linear1')(embedded_inputs)

#         return nn.sigmoid(logits),None
    

class MulticlassEmbeddingOnlyModel(nn.Module):
    num_classes: int
    def setup(self):
        self.linear1 = nn.Dense(features=self.num_classes, kernel_init=nn.initializers.glorot_normal())#,bias_init=nn.initializers.normal(stddev=0.1))
        self.dropout = nn.Dropout(rate=0.5)
    
    def __call__(self, x_embed,train=False):  # embedded_inputs: (batch, embed_dim) 
        # Dropout + classifier head
        x_embed = nn.relu(x_embed)
        x_drop = self.dropout(x_embed, deterministic=not train)
        logits = self.linear1(x_drop)             # (batch, num_classes)

        return logits, x_embed
    
    
    


class BagOfWordsClassifierMultiClass(nn.Module):
    num_classes: int
    vocabulary_size: int = 20002
    embedding_size: int = 50
    
    
    def setup(self):
        self.linear1 = nn.Dense(features=self.num_classes, kernel_init=nn.initializers.glorot_normal())#,bias_init=nn.initializers.normal(stddev=0.1))
        self.embed = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size,
                              embedding_init=nn.initializers.glorot_normal())
        self.dropout = nn.Dropout(rate=0.8)

    def __call__(self, x, train=False):
        
        x = np.asarray(x)  # Ensure x is a numpy array
        # debug.print('x: shape: {sx},{maxx},{minx}',sx=np.shape(x),maxx = np.max(x),minx = np.min(x))
        # x: (batch_size, seq_length), assumed padded with 0
        mask = (x != 0).astype(np.float32)  # Padding mask

        # debug.print('mask: {x}',x=np.shape(mask))
        embeddings = self.embed(x)  # (batch_size, seq_length, embedding_size)
        # debug.print('embeddings: {x}',x=np.shape(embeddings))
        embeddings = embeddings * mask[..., None]
        # debug.print('embeddings2: {x}',x=np.shape(embeddings))
        summed = np.sum(embeddings, axis=1)
        # debug.print('summed: {x}',x=np.shape(summed))
        lengths = np.sum(mask, axis=1, keepdims=True)
        # debug.print('shape of input: {y}\n lengths: {x}',x=np.shape(lengths),y = np.shape(x))
        x = summed / np.maximum(lengths, 1) 
        # debug.print('x after avg: {x}',x=np.shape(x))
        x_drop = self.dropout(x, deterministic=not train)
        logits = self.linear1(x_drop)
        # debug.print('logits:{x}',x=np.shape(logits))
        # debug.print('logits:{x}',x=np.isnan(logits).any())
        return logits, x

import flax.linen as nn
import jax.numpy as jnp

class MNISTClassifier(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 64   # size of intermediate embedding

    def setup(self):
        # Dense layer maps raw 784-dim pixels into hidden embedding
        self.embed_layer = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.glorot_normal()
        )
        self.dropout = nn.Dropout(rate=0.5)

        # Classifier head
        self.linear1 = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.glorot_normal()
        )

    def __call__(self, x, train=False):
        """
        x shape: (batch, 28, 28) with values in [0, 255]
        """
        x = jnp.array(x)  # Ensure x is a JAX array
        # Normalize pixel intensities to [0, 1]
        x = x.astype(jnp.float32) / 255.0

        # Flatten into vector
        x = x.reshape(x.shape[0], -1)  # (batch, 784)

        # Project into embedding space
        x_embed = self.embed_layer(x)             # (batch, hidden_dim)
        x_embed = nn.relu(x_embed)                # nonlinearity

        # Dropout + classifier head
        x_drop = self.dropout(x_embed, deterministic=not train)
        logits = self.linear1(x_drop)             # (batch, num_classes)

        return logits, x_embed

class MNISTConvClassifier(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 64   # size of intermediate embedding

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3,3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3,3))
        self.fc = nn.Dense(features=128)
        self.linear1 = nn.Dense(features=self.num_classes)
    
    def __call__(self, x, train=False):
        x = jnp.array(x)  # Ensure x is a JAX array
        # Normalize pixel intensities to [0, 1]
        x = x.astype(jnp.float32) / 255.0

        # x = x / 255.0
        x = x[..., None]  # add channel dim
        x = nn.relu(self.conv1(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.relu(self.conv2(x))
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(self.fc(x))
        logits = self.linear1(x)
        return logits, x



# Define the CNN architecture as a Flax nn.Module
class MNISTConvClassifierGemini(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128  # size of the high-level feature embedding

    def setup(self):
        # --- Feature Extraction Layers (CNN) ---
        # 1. First Convolutional Block
        self.conv1 = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=nn.initializers.lecun_normal()
        )
        # 2. Second Convolutional Block
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            kernel_init=nn.initializers.lecun_normal()
        )

        # --- Embedding Layer ---
        # This Dense layer maps the flattened feature maps into the final embedding vector.
        self.feature_layer = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.glorot_normal()
        )
        
        # --- Regularization and Classifier Head ---
        self.dropout = nn.Dropout(rate=0.4)
        
        # Classifier head: Maps the embedding to the final class logits
        self.linear1 = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.glorot_normal()
        )

    def __call__(self, x: jnp.ndarray, train: bool = False):
        """
        x shape: (batch, 28, 28) or (batch, 784) with values in [0, 255]
        """
        # --- 1. Preprocessing ---
        x = jnp.array(x)
        
        # Handle flattened input (B, 784) by reshaping to (B, 28, 28)
        if x.ndim == 2 and x.shape[-1] == 784:
            batch_size = x.shape[0]
            # Ensure the input is 28x28 before convolutions
            x = x.reshape((batch_size, 28, 28))

        # Normalize pixel intensities to [0, 1]
        # x = x.astype(jnp.float32) / 255.0
        
        # Add channel dimension (1 for grayscale). Input should now be (B, 28, 28).
        # Reshape from (B, 28, 28) to (B, 28, 28, 1)
        x = jnp.expand_dims(x, axis=-1)

        # --- 2. Feature Extraction (CNN) ---
        
        # Conv 1 -> ReLU -> MaxPool
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Conv 2 -> ReLU -> MaxPool
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Flatten feature maps into vector
        x = x.reshape((x.shape[0], -1))  # (batch, flattened_size)

        # --- 3. Embedding Generation ---
        # Project into the high-level embedding space (your requested point of interest)
        features = self.feature_layer(x)
        x_embed = nn.relu(features)  # Apply non-linearity to the embedding

        # --- 4. Classification ---
        # Dropout + classifier head
        x_drop = self.dropout(x_embed, deterministic=not train)
        logits = self.linear1(x_drop)  # (batch, num_classes)

        # Return both the logits and the meaningful feature embedding
        return logits, x_embed

class MNISTConvClassifierGemini2(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128  # final Dense embedding before classifier
    dropout_rate: float = 0.4

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = jnp.array(x)
        # Reshape flattened input
        if x.ndim == 2 and x.shape[-1] == 784:
            x = x.reshape((x.shape[0], 28, 28, 1))
        else:
            x = jnp.expand_dims(x, axis=-1)  # (B,28,28,1)
        
        # --- First conv block ---
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # --- Second conv block ---
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5,5), strides=(2,2), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        # --- Dense embedding ---
        x = x.reshape((x.shape[0], -1))  # flatten
        embedding = nn.Dense(features=self.hidden_dim)(x)
        embedding = nn.relu(embedding)
        embedding = nn.Dropout(rate=self.dropout_rate)(embedding, deterministic=not train)

        # --- Classifier head ---
        logits = nn.Dense(features=self.num_classes,name='linear1')(embedding)

        return logits, embedding

class SimpleMNISTConv(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128  # final Dense embedding before classifier
    

    @nn.compact
    def __call__(self, x, train=False):
        # x = x.reshape((x.shape[0],28,28,1))
        # Reshape flattened input
        if x.ndim == 2 and x.shape[-1] == 784:
            x = x.reshape((x.shape[0], 28, 28, 1))
        else:
            x = jnp.expand_dims(x, axis=-1)  # (B,28,28,1)
        
        x = nn.Conv(features=64, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2,2))
        x = x.reshape((x.shape[0], -1))
        embedding = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(embedding)
        logits = nn.Dense(self.num_classes,name='linear1')(x)
        
        return logits, embedding


import jax.numpy as jnp
from flax import linen as nn

class ImprovedMNISTConv(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 64  # Reduced to prevent overfitting on small data
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train=False):
        # 1. Flexible Input Handling
        if x.ndim == 2:
            x = x.reshape((x.shape[0], 28, 28, 1))
        elif x.ndim == 3:
            x = jnp.expand_dims(x, axis=-1)

        # 2. Feature Extraction with GroupNorm
        # GroupNorm is better than BatchNorm for small batches/small data
        x = nn.Conv(features=32, kernel_size=(3, 3), use_bias=False)(x)
        x = nn.GroupNorm(num_groups=4)(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), use_bias=False)(x)
        x = nn.GroupNorm(num_groups=4)(x)
        x = nn.relu(x)

        # 3. Global Pooling (Generalization Key)
        # Instead of flattening a huge tensor, we pool to preserve spatial invariance
        x = jnp.mean(x, axis=(1, 2)) 

        # 4. Bottleneck Embedding
        # Dropout here helps ensure the model doesn't rely on single 'lucky' features
        embedding = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(embedding)
        x = nn.relu(x)
        
        logits = nn.Dense(self.num_classes, name='classifier')(x)
        
        return logits, embedding

class RobustMNISTConv(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, x, train: bool = False):
        # 1. Input Normalization (Crucial for small data)
        # Ensure pixel values are roughly -1 to 1 or 0 to 1
        if x.ndim == 2:
            x = x.reshape((x.shape[0], 28, 28, 1))
        
        # 2. Convolutional Block 
        # Added explicit kernel_init for faster convergence
        x = nn.Conv(features=32, kernel_size=(3, 3), 
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.LayerNorm()(x) # LayerNorm is often more stable than GroupNorm in JAX
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = nn.Conv(features=64, kernel_size=(3, 3), 
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        # 3. Transition to Dense
        x = x.reshape((x.shape[0], -1)) 
        
        # 4. The Embedding Layer (Your counterfactual surface)
        embedding = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(embedding)
        
        # Dropout can cause 'failure to learn' if used too early or with small data
        if train:
            x = nn.Dropout(rate=0.2, deterministic=False)(x)
            
        logits = nn.Dense(self.num_classes, kernel_init=nn.initializers.glorot_normal())(x)
        
        return logits, embedding

class OptimizedMNISTConv(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, train=False):
        # 1. Precise Reshaping
        if x.ndim == 2:
            x = x.reshape((x.shape[0], 28, 28, 1))
        
        # 2. Reverting to your specific Conv stack (it clearly works for your data)
        # We add 'Same' padding to prevent edge-information loss
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME', name='conv1')(x)
        x = nn.relu(x)
        
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME', name='conv2')(x)
        x = nn.relu(x)
        
        # Pooling can sometimes be too aggressive on small data
        x = nn.max_pool(x, (2, 2))
        
        # 3. Flattening (Restoring the high-dim connection)
        x = x.reshape((x.shape[0], -1))
        
        # 4. The Embedding Layer
        # We use a 'Leaky ReLU' here. This ensures that even for counterfactual 
        # directions that move 'away' from the data, the gradient doesn't die.
        embedding = nn.Dense(self.hidden_dim, name='fc_embed')(x)
        x = nn.leaky_relu(embedding, negative_slope=0.01)
        
        # 5. Final Classification
        logits = nn.Dense(self.num_classes, name='linear1')(x)
        
        return logits, embedding
    
import jax.numpy as jnp
from flax import linen as nn

class SurpassingMNIST(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, x, train=False):
        if x.ndim == 2: x = x.reshape((x.shape[0], 28, 28, 1))
        
        # Path A: Fine detail (3x3) - similar to your original
        a = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        a = nn.relu(a)
        
        # Path B: Structural context (5x5) - better generalization
        b = nn.Conv(features=32, kernel_size=(5, 5), padding='SAME')(x)
        b = nn.relu(b)
        
        # Merge the paths
        x = jnp.concatenate([a, b], axis=-1)
        
        # Second Stage: Spatial contraction
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        # The Counterfactual Surface (Bottleneck)
        # Use a "Squeeze-and-Excitation" style weighting to let the model 
        # ignore noisy features on small data
        embedding = nn.Dense(self.hidden_dim)(x)
        
        # SWISH is smoother than ReLU and usually provides better counterfactual 
        # gradients because it is non-monotonic and differentiable everywhere.
        x = nn.swish(embedding)
        
        logits = nn.Dense(self.num_classes, name='linear1')(x)
        return logits, embedding
    
class ManifoldRobustConv(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128
    dropout_rate: float = 0.3 # Higher dropout forces generalization

    @nn.compact
    def __call__(self, x, train=False):
        if x.ndim == 2: x = x.reshape((x.shape[0], 28, 28, 1))
        
        # Block 1: Feature Extraction
        x = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (2, 2))
        
        # Block 2: Spatial Hierarchy
        x = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.swish(x)
        x = nn.max_pool(x, (2, 2))
        
        x = x.reshape((x.shape[0], -1))
        
        # THE CRITICAL PART: The Embedding Surface
        # Adding Dropout here prevents 100% training accuracy 'tunnel vision'
        embedding = nn.Dense(self.hidden_dim)(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(embedding)
        x = nn.swish(x)
        
        logits = nn.Dense(self.num_classes, name='linear1')(x)
        return logits, embedding
    
import jax.numpy as jnp
import flax.linen as nn

class MNISTViTClassifier(nn.Module):
    num_classes: int = 10
    hidden_dim: int = 128  # size of the high-level feature embedding
    patch_size: int = 7
    emb_dim: int = 64
    num_layers: int = 4
    num_heads: int = 4
    mlp_dim: int = 128
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1


    def setup(self):
        # Dense projection for the final embedding
        self.feature_layer = nn.Dense(
            features=self.hidden_dim,
            kernel_init=nn.initializers.glorot_normal()
        )
        self.dropout = nn.Dropout(rate=self.dropout_rate)
        self.linear1 = nn.Dense(
            features=self.num_classes,
            kernel_init=nn.initializers.glorot_normal()
        )
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False):
        """
        x: (batch, 28, 28) or (batch, 784) with values in [0, 255]
        Returns: (logits, features)
        """
        # --- 1. Preprocessing ---
        x = jnp.array(x)
        if x.ndim == 2 and x.shape[-1] == 784:
            batch_size = x.shape[0]
            x = x.reshape((batch_size, 28, 28))
        x = x.astype(jnp.float32) / 255.0
        x = jnp.expand_dims(x, axis=-1)  # (B,28,28,1)

        # --- 2. Patch embedding ---
        patch_h = patch_w = self.patch_size
        n_h, n_w = x.shape[1] // patch_h, x.shape[2] // patch_w
        n_patches = n_h * n_w

        x = nn.Conv(
            features=self.emb_dim,
            kernel_size=(patch_h, patch_w),
            strides=(patch_h, patch_w),
            name="patch_embedding"
        )(x)  # (B, n_h, n_w, emb_dim)
        x = x.reshape((x.shape[0], -1, self.emb_dim))  # (B, n_patches, emb_dim)

        # cls token
        cls = self.param('cls', nn.initializers.zeros, (1, 1, self.emb_dim))
        cls = jnp.tile(cls, [x.shape[0], 1, 1])
        x = jnp.concatenate([cls, x], axis=1)  # (B, n_patches+1, emb_dim)

        # pos embedding
        pos_emb = self.param(
            'pos_embedding',
            nn.initializers.normal(stddev=0.02),
            (1, n_patches + 1, self.emb_dim)
        )
        x = x + pos_emb
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)

        # --- 3. Transformer encoder ---
        for i in range(self.num_layers):
            x = nn.LayerNorm()(x)
            attn = nn.SelfAttention(
                num_heads=self.num_heads,
                dropout_rate=self.attention_dropout_rate,
                deterministic=not train,
                name=f"attn_{i}"
            )(x)
            attn = nn.Dropout(self.dropout_rate)(attn, deterministic=not train)
            x = x + attn

            y = nn.LayerNorm()(x)
            y = nn.Dense(self.mlp_dim)(y)
            y = nn.gelu(y)
            y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)
            y = nn.Dense(self.emb_dim)(y)
            y = nn.Dropout(self.dropout_rate)(y, deterministic=not train)
            x = x + y

        x = nn.LayerNorm()(x)
        cls_token = x[:, 0]  # (B, emb_dim)

        # --- 4. Embedding + classification ---
        features = self.feature_layer(cls_token)
        x_embed = nn.relu(features)
        x_drop = self.dropout(x_embed, deterministic=not train)
        logits = self.linear1(x_drop)

        return logits, features

# # Define the CNN architecture as a Flax nn.Module
# class MNISTConvClassifierGemini(nn.Module):
#     num_classes: int = 10
#     hidden_dim: int = 128  # size of the high-level feature embedding

#     def setup(self):
#         # --- Feature Extraction Layers (CNN) ---
#         # 1. First Convolutional Block
#         self.conv1 = nn.Conv(
#             features=32,
#             kernel_size=(3, 3),
#             kernel_init=nn.initializers.lecun_normal()
#         )
#         # 2. Second Convolutional Block
#         self.conv2 = nn.Conv(
#             features=64,
#             kernel_size=(3, 3),
#             kernel_init=nn.initializers.lecun_normal()
#         )

#         # --- Embedding Layer ---
#         # This Dense layer maps the flattened feature maps into the final embedding vector.
#         self.feature_layer = nn.Dense(
#             features=self.hidden_dim,
#             kernel_init=nn.initializers.glorot_normal()
#         )
        
#         # --- Regularization and Classifier Head ---
#         self.dropout = nn.Dropout(rate=0.5)
        
#         # Classifier head: Maps the embedding to the final class logits
#         self.classifier_head = nn.Dense(
#             features=self.num_classes,
#             kernel_init=nn.initializers.glorot_normal()
#         )

#     def __call__(self, x: jnp.ndarray, train: bool = False):
#         """
#         x shape: (batch, 28, 28) with values in [0, 255]
#         """
#         print("SHAPE START:",np.shape(x))
#         # --- 1. Preprocessing ---
#         # Normalize pixel intensities to [0, 1] and add channel dimension (1 for grayscale)
#         x = jnp.array(x)
#         x = x.astype(jnp.float32) / 255.0
#         # Reshape from (B, 28, 28) to (B, 28, 28, 1)
#         x = jnp.expand_dims(x, axis=-1)

#         # --- 2. Feature Extraction (CNN) ---
        
#         # Conv 1 -> ReLU -> MaxPool
#         x = self.conv1(x)
#         x = nn.relu(x)
#         x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
#         # Conv 2 -> ReLU -> MaxPool
#         x = self.conv2(x)
#         x = nn.relu(x)
#         x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
#         print(np.shape(x))
#         # Flatten feature maps into vector
#         x = x.reshape((x.shape[0], -1))  # (batch, flattened_size)

#         # --- 3. Embedding Generation ---
#         # Project into the high-level embedding space (your requested point of interest)
#         x_embed = self.feature_layer(x)
#         x_embed = nn.relu(x_embed)  # Apply non-linearity to the embedding

#         # --- 4. Classification ---
#         # Dropout + classifier head
#         x_drop = self.dropout(x_embed, deterministic=not train)
#         logits = self.classifier_head(x_drop)  # (batch, num_classes)

#         # Return both the logits and the meaningful feature embedding
#         return logits, x_embed


class BagOfWordsClassifier2Layer(nn.Module):
    vocabulary_size: int = 20002
    embedding_size: int = 50
    
    def setup(self):
        self.linear1 = nn.Dense(features=32, kernel_init=nn.initializers.glorot_normal())#,bias_init=nn.initializers.normal(stddev=0.1))
        self.linear2 = nn.Dense(features=1, kernel_init=nn.initializers.glorot_normal())#,bias_init=nn.initializers.normal(stddev=0.1))
        self.embed = nn.Embed(num_embeddings=self.vocabulary_size, features=self.embedding_size,
                              embedding_init=nn.initializers.glorot_normal())
        self.dropout = nn.Dropout(rate=0.9)

    def __call__(self, x, train=False):
        
        x = np.asarray(x)  # Ensure x is a numpy array
        # debug.print('x: shape: {sx},{maxx},{minx}',sx=np.shape(x),maxx = np.max(x),minx = np.min(x))
        # x: (batch_size, seq_length), assumed padded with 0
        mask = (x != 0).astype(np.float32)  # Padding mask

        # debug.print('mask: {x}',x=np.mean(mask))
        embeddings = self.embed(x)  # (batch_size, seq_length, embedding_size)
        # debug.print('embeddings: {x}',x=np.mean(embeddings))
        embeddings = embeddings * mask[..., None]
        # debug.print('embeddings2: {x}',x=np.mean(embeddings))
        summed = np.sum(embeddings, axis=1)

        lengths = np.sum(mask, axis=1, keepdims=True)
        # debug.print('shape of input: {y}\n lengths: {x}',x=np.shape(lengths),y = np.shape(x))
        x = summed / np.maximum(lengths, 1) 

        x_drop = self.dropout(x, deterministic=not train)
        x_drop = nn.relu(self.linear1(x_drop))
        x_drop = self.dropout(x_drop, deterministic=not train)
        logits = self.linear2(x_drop)
        return logits.squeeze(axis=-1), x


class SentimentModel(nn.Module):
    vocab_size: int = 20000
    embed_size: int = 50

    @nn.compact
    def __call__(self, x, mask):
        # x: (batch_size, 32) token indices
        x = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_size)(x)
        # x: (batch_size, 32, 50)
        # Average over tokens, ignoring padding
        x = np.sum(x * mask[..., None], axis=1) / np.sum(mask, axis=1, keepdims=True)
        # x: (batch_size, 50)
        x = nn.Dense(1)(x)
        x = nn.sigmoid(x)
        return x

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

import flax.linen as nn
from flax.linen.initializers import glorot_normal

class MNIST_v1(nn.Module):
    num_hidden: int   # number of hidden neurons
    num_outputs: int = 10  # 10 digits (0–9)

    def setup(self):
        # Layers
        self.linear1 = nn.Dense(features=self.num_hidden, kernel_init=glorot_normal())
        self.linear2 = nn.Dense(features=self.num_outputs, kernel_init=glorot_normal())
        self.dropout = nn.Dropout(rate=0.1)

    def __call__(self, x, train: bool = False):
        # x shape: (batch_size, 784) for MNIST
        x = self.linear1(x)
        x = nn.relu(x)
        # Dropout only if training
        # x = self.dropout(x, deterministic=not train)
        x = self.linear2(x)  # logits, shape (batch_size, 10)

        return x, None  # logits + dummy placeholder


custom_models = {'simple':SimpleClassifier,
                 'simple_v2':SimpleClassifier_v2,
                 'multiclass':MultiClassClassifier, 
                 'bag_of_words':BagOfWordsClassifierSimple,
                 'mnist':MNISTClassifier,
                 'mnist_conv':MNISTConvClassifier,
                 'simple_mnist_conv':SimpleMNISTConv,
                 'improved_mnist_conv':ImprovedMNISTConv,
                 'robust_mnist_conv':RobustMNISTConv,
                 'manifold_robust_mnist_conv':ManifoldRobustConv,
                 'optimized_mnist_conv':OptimizedMNISTConv,
                 'finalform_mnist_conv':SurpassingMNIST,
                 'mnist_conv_gemini':MNISTConvClassifierGemini2,
                 'mnist_vit':MNISTViTClassifier,
                 'multiclass_bag_of_words':BagOfWordsClassifierMultiClass,
                 'gradient_supervision':GSPaper, 
                 'GPTattempt':TextClassifierHard}