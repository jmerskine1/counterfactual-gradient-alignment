# base imports
import numpy as np 
import pandas as pd
from functools import partial
import pickle
import warnings
import os

# preprocessing
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

import re

# NLP packages
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import scipy

max_length = 16

"""
Data Preprocessing Functions
"""
stemmer = PorterStemmer()
lemma = WordNetLemmatizer()

def tokenize(text,vocabulary,max_length = 32):
    # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
    # tokens = [word for word in word_tokenize(text.lower()) if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    tokens = [word for word in text.lower() if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    # tokens = [word for word in word_tokenize(text.lower())]# if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    indexed_tokens = [vocabulary.get(token, -1)+2 for token in tokens]  # 1 for unknown words

    # Trim to 32 tokens or pad with -1s if shorter
    if len(indexed_tokens) < max_length:
        indexed_tokens.extend([0] * (max_length - len(indexed_tokens)))  # Padding with -1
    else:
        indexed_tokens = indexed_tokens[:max_length]  # Trim to n_tokens=max_length

    
    # return [vocabulary.get(token, -1) for token in tokens]  # -1 for unknown words
    return indexed_tokens

def token2index(tokens,vocabulary,max_length = 32):
    # my text was unicode so I had to use the unicode-specific translate function. If your documents are strings, you will need to use a different `translate` function here. `Translated` here just does search-replace. See the trans_table: any matching character in the set is replaced with `None`
    # tokens = [word for word in word_tokenize(text.lower()) if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    
    # tokens = [word for word in word_tokenize(text.lower())]# if len(word) > 1] #if len(word) > 1 because I only want to retain words that are at least two characters before stemming, although I can't think of any such words that are not also stopwords
    indexed_tokens = [vocabulary.get(token, -1)+2 for token in tokens]  # 1 for unknown words

    # Trim to 32 tokens or pad with -1s if shorter
    if len(indexed_tokens) < max_length:
        indexed_tokens.extend([0] * (max_length - len(indexed_tokens)))  # Padding with -1
    else:
        indexed_tokens = indexed_tokens[:max_length]  # Trim to n_tokens=max_length

    
    # return [vocabulary.get(token, -1) for token in tokens]  # -1 for unknown words
    return indexed_tokens

def assert_even_length_array(arr):
    if len(arr) % 2 != 0:
        raise AssertionError("Array does not have an even number of elements: {}".format(len(arr)))

def check_order(array):
    prev_entry = None
    ilist = []  
    for i, entry in enumerate(array):
        if entry == prev_entry:
            ilist.append(i)
    
        prev_entry = entry

    return ilist

def sort_data_cfs(array,ilist):
    # given a list of indices, swap the index and the following index?
    for i in ilist:
        temp1 = array[i]
        temp2 = array[i+1]

        array.iloc[i] = temp2
        array.iloc[i+1] = temp1

    return array

def sort_label_cfs(array,ilist):
    # given a list of indices, swap the index and the following index?
    for i in ilist:
        temp1 = array[i]
        temp2 = array[i+1]

        array[i] = temp2
        array[i+1] = temp1

    return array


def get_vector(vec1,vec2):
    
    d_vec = vec2 - vec1
    
    if np.sum(d_vec) == 0:
        
        warnings.warn("A text vector and its' counterfactual vector have the same value. This is probably not right.")
        
    if isinstance(d_vec, scipy.sparse.csr_matrix):  # or any other sparse matrix type
        mag = scipy.linalg.norm(d_vec.todense())

    elif isinstance(d_vec, np.ndarray):
        mag = np.linalg.norm(d_vec)
        # Perform operations specific to dense arrays
    else:
        print("Unknown vector type")
        mag = np.linalg.norm(d_vec)
    
    if mag!=0:
        return d_vec/mag        
    else:
        return np.zeros((np.shape(vec1)[0]))
    # return [d_vec/mag]



## cleaning the text

def cleantext(text):
    # removing the "\"
    text = re.sub("'\''","",text)
    
    # removing special symbols
    text = re.sub("[^ a-zA-Z]","",text)
    
    # removing the whitespaces
    text = ' '.join(text.split())
    
    # convert text to lowercase    
    text = text.lower()
    
    return text

# removing the stopwords
stop_words = set(stopwords.words('english'))
mod_stop_words = set(stopwords.words('english')) - {'not', 'but'}
# stop_words = stop_words - set(['dont','do'])
# def removestopwords(text):
    
#     removedstopword = [word for word in text.split() if word not in mod_stop_words]
#     return ' '.join(removedstopword)

def removestopwords(text):
    
    removedstopword = [word for word in text if word not in mod_stop_words]
    return removedstopword

#Removing the html strips 
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

#Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

#Removing Emails
def remove_Emails(text):
    pattern=r'\S+@\S+'
    text=re.sub(pattern,'',text)
    return text

#Removing URLS
def remove_URLS(text):
    pattern=r'http\S+'
    text=re.sub(pattern,'',text)
    return text

def remove_special_characters(text, remove_digits=True):
    pattern=r'[^a-zA-z0-9\s]'
    text=re.sub(pattern,' ',text)
    return text

#Removing numbers
def remove_numbers(text):
    pattern = r'\d+'
    text = re.sub(pattern, '', text)
    return text

def lematizing(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def stemming(sentence):
    
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence+=stem
        stemmed_sentence+=" "
        
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence



# def process_data(data):
#     data=data.apply(lambda x:strip_html(x))
#     data=data.apply(remove_between_square_brackets)
#     data = data.apply(lambda x:cleantext(x))
    
#     data=data.apply(remove_special_characters)

#     data = data.apply(lambda x: word_tokenize(x))
#     print(data)
    
    
    
#     data=data.apply(remove_URLS)
#     data=data.apply(remove_Emails)
    
#     data=data.apply(remove_numbers)
    

    
#     data = data.apply(lambda x:removestopwords(x))
#     # data = data.apply(lambda x: lematizing(x))
#     data = data.apply(lambda text:stemming(text))
    
#     # data = [text_to_vector(txt) for txt in data]
    
#     return data

def process_data(data):
    # Step 1: clean text (string level)
    data = data.apply(lambda x: strip_html(x))
    data = data.apply(remove_between_square_brackets)
    data = data.apply(lambda x: cleantext(x))
    data = data.apply(remove_special_characters)
    data = data.apply(remove_URLS)
    data = data.apply(remove_Emails)
    data = data.apply(remove_numbers)
    
    # Step 2: tokenize
    data = data.apply(lambda x: word_tokenize(x.lower()))
    
    # Step 3: remove stopwords and stem (token level)
    data = data.apply(lambda tokens: [stemming(token) for token in removestopwords(tokens)])

    return data




def gen_knowledge(data,label,cf=False):
    """
    Takes sequences of pairs of counterfactuals and 
    returns just the originals, with the counterfactual 
    included as an annotation.

    If cf=True, returns the counterfactuals and omits the originals
    """
    
    vectors_in = data['vector']
    text_in = data['text']
    # sparse_array = sp.csr_matrix((0,dlength))

    text_out=[]
    vectors_out = []
    labels = []
    # vectors = np.empty((0,1,dlength))
    cf_text = []
    cf_labels = []
    cf_vectors = []
    
    for i in range(int(np.shape(vectors_in)[0]/2)):
        
        
        if cf:
            vec1,vec2 = vectors_in[i * 2 + 1],vectors_in[i * 2]  # Get the data from the current dataset slice
            label_out = label[i*2+1]
            text = text_in[i * 2 + 1]
            _cf_text=text_in[i * 2]
        else:
            vec1,vec2 = vectors_in[i * 2],vectors_in[i * 2 + 1]  # Get the data from the current dataset slice                
            label_out = label[i*2]
            text = text_in[i * 2]
            _cf_text = text_in[i * 2 + 1]
            
        text_out.append(text)
        labels.append(label_out)
        vectors_out.append(vec1)
    
        cf_vectors.append([vec2])
        cf_labels.append([1 - label_out])
        cf_text.append([_cf_text])
    
    return {'vector':vectors_out,'text':text_out},labels,cf_vectors, cf_labels,cf_text



def compile_K(data,label, paired=False,cf=False,int_out=False):
    
    if paired:
        
        if np.shape(data['vector'])[0]%2 != 0:
            raise ValueError("Can't generate paired counterfactuals with an uneven number of samples")
        
        data,label,vector,labels,cf_text = gen_knowledge(data,label,cf=cf)
        
        # labels = [1 - l for l in label]
        
    else:
        warnings.warn("Non-paired data just creates blanks atm")
        
        vector = [[] for _ in range(np.shape(data['vector'])[0])]
        cf_text = [[] for _ in range(np.shape(data['vector'])[0])]

        labels = [[]]*len(vector)
    
    # labels = np.expand_dims(labels, axis=1)

    magnitude = np.ones(len(vector))
    magnitude = np.expand_dims(magnitude, axis=1)
    

    if int_out:
        for i in range(np.shape(vector)[0]):
            vector[i] = vector[i].astype(int)

    n_samples = len(data['text'])
    print(f'Returning {n_samples} samples with {len(vector)} counterfactuals')
    
    return {'text':data['text'],
            'X':data['vector'],
            'Y':label,
            'K':{'vector':vector,'label':labels,'magnitude':magnitude, 'text':cf_text}}

# Get the directory of the current Python file
base_dir = os.path.dirname(os.path.abspath(__file__))

# this is just the counterfactually augmented dat from kaushik
train_data = pd.read_table(os.path.join(base_dir,"data/paired/train_paired.tsv"))
test_data = pd.read_table(os.path.join(base_dir,"data/paired/test_paired.tsv"))
dev_data = pd.read_table(os.path.join(base_dir,"data/paired/dev_paired.tsv"))
combined_data = pd.concat([train_data, test_data, dev_data], ignore_index=True)

dev_paired = dev_data['Text']
dev_label = np.array(dev_data['Sentiment'].map({'Positive': 1, 'Negative': 0}))

test_paired = test_data['Text']
test_label = np.array(test_data['Sentiment'].map({'Positive': 1, 'Negative': 0}))
 
train_paired = train_data['Text']
train_label = np.array(train_data['Sentiment'].map({'Positive': 1, 'Negative': 0}))

# dev_paired = process_data(dev_paired)
# test_paired = process_data(test_paired)
# train_paired = process_data(train_paired)

# vectorizer = CountVectorizer(max_features=20000)
# vectorizer.fit(train_paired)
# # Get the vocabulary mapping (word to integer index)
# vocabulary = vectorizer.vocabulary_


# tfidf_train = [tokenize(text,vocabulary,max_length=max_length) for text in train_paired]
# tfidf_test = [tokenize(text,vocabulary,max_length=max_length) for text in test_paired]
# tfidf_dev = [tokenize(text,vocabulary,max_length=max_length) for text in dev_paired]

dev_paired = process_data(dev_paired)
test_paired = process_data(test_paired)
train_paired = process_data(train_paired)



data_for_vectorizer = train_paired.apply(lambda tokens: " ".join(tokens))
vectorizer = CountVectorizer(max_features=20000)
vectorizer.fit(data_for_vectorizer)
# Get the vocabulary mapping (word to integer index)
vocabulary = vectorizer.vocabulary_

tfidf_train = [token2index(text,vocabulary,max_length=max_length) for text in train_paired]
tfidf_test = [token2index(text,vocabulary,max_length=max_length) for text in test_paired]
tfidf_dev = [token2index(text,vocabulary,max_length=max_length) for text in dev_paired]

"""
########################################################################################################################
Save embeddings
########################################################################################################################
"""
# Stripping html from unprocessed text, just to clean it up
print('\ntrain_Data')
cf_X = {'text':[strip_html(txt) for txt in train_data['Text']],'vector':np.array(tfidf_train)}
cf_train={'original': compile_K(cf_X,train_label,paired=True ),
         'modified': compile_K(cf_X,train_label,paired=True,cf=True)
         }

print('\ndev_Data')
cf_X = {'text':[strip_html(txt) for txt in dev_data['Text']],'vector':np.array(tfidf_dev)}
cf_dev ={'original': compile_K(cf_X,dev_label,paired=True),
         'modified': compile_K(cf_X,dev_label,paired=True,cf=True)
         }

print('\ntest_Data')
cf_X = {'text':[strip_html(txt) for txt in test_data['Text']],'vector':np.array(tfidf_test)}
cf_test={'original': compile_K(cf_X,test_label,paired=True),
         'modified': compile_K(cf_X,test_label,paired=True,cf=True)
         }


pickle_data = {'train':cf_train,'test':cf_test,'dev':cf_dev}

embedding_path = os.path.join(base_dir,f'data/integer_len{max_length}.pkl')
print(f"Saving to {embedding_path}")
with open(embedding_path, 'wb') as file:
    pickle.dump(pickle_data, file)