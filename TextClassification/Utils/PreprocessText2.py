# conda install nltk and do nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
# do nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
from string import punctuation
from tqdm import tqdm
'''for the following model see: do pip install contractions 
(see https://github.com/kootenpv/contractions) and do pip install textsearch
since, there is no nltk module for contractions.

1. pip install textsearch
2. pip install contractions
3. nltk.download('stopwords')
4. nltk.download('wordnet')'''
import contractions
import numpy as np
import multiprocessing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
'''
A generic function to process all textual data
'''
F_PTH = '<PathToGloveEmbeddings>'
FIL_N = 'glove.6B.200d.txt'

def __textOperations(text):
    lmtzr = WordNetLemmatizer()
    # start by removing contractions
    text = contractions.fix(text)
    # tokenizes the sentence by considering only alpha numeric characters
    tokens = word_tokenize(text)
    tokens = [x.lower() for x in tokens]
    tokens = [x for x in tokens if x not in punctuation]
    tokens = [lmtzr.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    doc_len = len(tokens)
    return (' '.join(tokens),doc_len)

def PurifyText(df,field):

    p = multiprocessing.Pool(8)
    mx_doc_len = 0
    vals = list(tqdm(p.imap(__textOperations, df[field]),total=len(df)))
    for v,l in vals:
        mx_doc_len = max(mx_doc_len,l)
    vals = [v[0] for v in vals]
    df[field+'_a'] = pd.DataFrame(vals)
    df.drop(field,axis=1,inplace=True)
    df.rename({field+'_a':field}, axis='columns',inplace=True)
    return (df,mx_doc_len)

class Embeddings:

    def __init__(self,typ):

        self.embeddings_index = dict()
        # size of the embeddings
        self.embed_sz = 100
        if typ == 'glove':
            self.__loadGlove()

    def __loadGlove(self):

        # we now start load the whole embedding into memory and selecting only those embeddings that correspond to our input vocabulary.

        f = open(F_PTH+FIL_N)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        self.embed_sz = len(self.embeddings_index[word])
        f.close()

    def getEmbeddings(self,X,mx_len):

        '''
        :param X: the raw text after some pre-processing
        :param mx_len: maximum length of a document
        :return: padded documents and embeddings loaded from glove
        '''
        # begin by tokenizing, integer encoding, and padding text using keras pre-processing module
        # the number of time steps is the maximum length of a document (calculated across all documents)
        n_timesteps = mx_len
        t = Tokenizer()
        t.fit_on_texts(X)
        vocab_size = len(t.word_index) + 1
        encoded_docs = t.texts_to_sequences(X)
        padded_docs = pad_sequences(encoded_docs, maxlen=n_timesteps, padding='post')
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, self.embed_sz))
        for word, i in t.word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return(padded_docs,embedding_matrix,self.embed_sz,vocab_size)

