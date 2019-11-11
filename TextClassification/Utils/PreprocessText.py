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
import os
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile, datapath
from gensim.scripts.glove2word2vec import glove2word2vec
import logging
from os.path import expanduser
HOME = expanduser("~")
'''
A generic function to process all textual data
'''
LOAD_EMB = True
if LOAD_EMB:

    F_PTH = HOME+'/Documents/DataRepo/PretrainedEmbeddings/'
    keyed_vecs = 'keyed-6B-300.bin.gz'
    if os.path.exists(F_PTH+keyed_vecs):
        WORD_MODEL = KeyedVectors.load_word2vec_format(F_PTH+keyed_vecs, binary=True)
        print('loaded embedding:'+keyed_vecs)
    else:
        FIL_A = 'Glove/glove.6B/glove.6B.300d.txt'
        FIL_B = 'PatentToGlove-'
        glove_file = datapath(F_PTH + FIL_A)
        word2vec_glove_file = get_tmpfile('glove.to.word2vec.txt')
        glove2word2vec(glove_file, word2vec_glove_file)
        WORD_MODEL = KeyedVectors.load_word2vec_format(word2vec_glove_file)
        WORD_MODEL.save_word2vec_format(F_PTH+'keyed-6B-300.bin.gz', binary=True)
        print('loaded embedding:'+keyed_vecs)

class PurifyText:

    def __init__(self,typ):
        self.typ = typ

    def __MeanEmbeddingVectorizer(self,docs):
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


    def textOperations(self, text):
        lmtzr = WordNetLemmatizer()
        # start by removing contractions
        text = contractions.fix(text)
        # tokenizes the sentence by considering only alpha numeric characters
        tokens = word_tokenize(text)
        tokens = [x.lower() for x in tokens]
        tokens = [x for x in tokens if x not in punctuation]
        tokens = [lmtzr.lemmatize(x) for x in tokens]
        tokens = [x for x in tokens if x not in stopwords.words('english')]
        if self.typ == 'word2vec averaging':
            tokens = self.__MeanEmbeddingVectorizer(tokens)
            doc_len = len(tokens)
            return (tokens, doc_len)
        doc_len = len(tokens)
        return (' '.join(tokens), doc_len)


    def getText(self, df, field):
        cpus = multiprocessing.cpu_count()
        p = multiprocessing.Pool(cpus)
        mx_doc_len = 0
        vals = list(tqdm(p.imap(self.textOperations, df[field]), total=len(df)))

        for v, l in vals:
            mx_doc_len = max(mx_doc_len, l)
        vals = [v[0] for v in vals]
        # if self.typ == 'word2vec averaging':
        #     return (np.array(vals), mx_doc_len)

        # df[field + '_a'] = pd.DataFrame(vals)
        # df.drop(field, axis=1, inplace=True)
        # df.rename({field + '_a': field}, axis='columns', inplace=True)
        return (np.array(vals), mx_doc_len)


class Embeddings:

    def __init__(self, typ):

        self.embeddings_index = dict()
        # size of the embeddings
        self.embed_sz = 100
        self.MAX_NB_WORDS = len(WORD_MODEL.wv.vocab)
        self.word_vectors = WORD_MODEL.wv
        self.nb_words = len(WORD_MODEL.wv.vocab)
        if typ == 'glove':
            self.__loadGlove()

    def __loadGlove(self):

        # we now start load the whole embedding into memory and selecting only those embeddings that correspond to our input vocabulary.

        f = open('glove.to.word2vec.txt')
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
        t = Tokenizer()
        t.fit_on_texts(X)
        vocab_size = len(t.word_index) + 1
        encoded_docs = t.texts_to_sequences(X)
        padded_docs = pad_sequences(encoded_docs, maxlen=mx_len, padding='post')
        # create a weight matrix for words in training docs
        wv_matrix = (np.random.rand(vocab_size, self.embed_sz) - 0.5) / 5.0
        for word, i in t.word_index.items():
            try:
                embedding_vector = self.word_vectors[word]
                # words not found in embedding index will be all-zeros.
                wv_matrix[i] = embedding_vector
            except:
                pass

        return (padded_docs, wv_matrix, self.embed_sz, vocab_size)
