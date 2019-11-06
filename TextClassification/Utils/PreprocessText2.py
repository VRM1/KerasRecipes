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
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile, datapath
from gensim.scripts.glove2word2vec import glove2word2vec
import logging

'''
A generic function to process all textual data
'''
F_PTH = '/home/vineeth/Documents/DataRepo/PretrainedEmbeddings/'
FIL_A = 'Glove/glove.6B/glove.6B.300d.txt'
FIL_B = 'PatentToGlove-'
glove_file = datapath(F_PTH + FIL_A)
word2vec_glove_file = get_tmpfile('glove.to.word2vec.txt')
glove2word2vec(glove_file, word2vec_glove_file)
WORD_MODEL = KeyedVectors.load_word2vec_format(word2vec_glove_file)


def MeanEmbeddingVectorizer(docs):
    # glove_mean_vec_tr = MeanEmbeddingVectorizer(glove_WORD_MODEL)
    # WORD_MODEL = glove_mean_vec_tr.transform(docs)

    vector_size = WORD_MODEL.wv.vector_size
    doc_vecs = []
    for word in docs:
        if word in WORD_MODEL.wv.vocab:
            doc_vecs.append(WORD_MODEL.wv.get_vector(word))

    if not doc_vecs:  # empty words
        doc_vecs.append(np.zeros(vector_size))

    return np.array(doc_vecs).mean(axis=0)


def __textOperations(text,typ):
    lmtzr = WordNetLemmatizer()
    # start by removing contractions
    text = contractions.fix(text)
    # tokenizes the sentence by considering only alpha numeric characters
    tokens = word_tokenize(text)
    tokens = [x.lower() for x in tokens]
    tokens = [x for x in tokens if x not in punctuation]
    tokens = [lmtzr.lemmatize(x) for x in tokens]
    tokens = [x for x in tokens if x not in stopwords.words('english')]
    if typ == 'word2vec averaging':
        tokens = MeanEmbeddingVectorizer(tokens)
    doc_len = len(tokens)
    return (' '.join(tokens), doc_len)


def PurifyText(df, field, typ):
    cpus = multiprocessing.cpu_count()
    p = multiprocessing.Pool(cpus)
    mx_doc_len = 0
    vals = list(tqdm(p.starmap(__textOperations, df[field], typ), total=len(df)))

    for v, l in vals:
        mx_doc_len = max(mx_doc_len, l)
    vals = [v[0] for v in vals]
    if typ == 'word2vec averaging':
        return (np.array(vals), mx_doc_len)

    df[field + '_a'] = pd.DataFrame(vals)
    df.drop(field, axis=1, inplace=True)
    df.rename({field + '_a': field}, axis='columns', inplace=True)
    return (vals, mx_doc_len)


class Embeddings:

    def __init__(self, typ):

        self.embeddings_index = dict()
        # size of the embeddings
        self.embed_sz = 100
        if typ == 'glove':
            self.__loadGlove()

    def __loadGlove(self):

        # we now start load the whole embedding into memory and selecting only those embeddings that correspond to our input vocabulary.

        f = open(F_PTH + FIL_N)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embeddings_index[word] = coefs
        self.embed_sz = len(self.embeddings_index[word])
        f.close()

    def getEmbeddings(self, X, mx_len):

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

        return (padded_docs, embedding_matrix, self.embed_sz, vocab_size)
