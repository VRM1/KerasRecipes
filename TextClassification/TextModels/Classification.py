
import numpy as np
from tqdm import tqdm
from keras.layers import Input, Dropout, Dense, Embedding, Flatten
from keras.layers import GRU, LSTM, CuDNNGRU, CuDNNLSTM
from keras.layers import Bidirectional, TimeDistributed
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
'''
References:
1. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
'''
# initialize file path of the embedding file and the name of the embeddings
F_PTH = '/Users/vineeth/OneDrive - Interdigital Communications Inc/DataRepo/PretrainedEmbeddings/Glove/glove.6B/'
FIL_N = 'glove.6B.200d.txt'
DEVICE = 'CPU'
if K.tensorflow_backend._get_available_gpus():
    DEVICE = 'GPU'

def SimpleLstm(n_outputs,vocab_sz,emb_sz,embedding_matrix,n_timesteps):

    x = Input(shape=(n_timesteps,))
    # start creating a simpleLSTM model
    e = Embedding(vocab_sz, emb_sz, weights=[embedding_matrix], input_length=n_timesteps, trainable=True)(x)
    if DEVICE == 'CPU':
        l_1 = Bidirectional(GRU(100,activation='relu'))(e)
    if DEVICE == 'GPU':
        l_1 = Bidirectional(CuDNNGRU(100, activation='relu',))(e)
    l_2 = Dropout(0.2)(l_1)
    l_3 = Dense(100, activation='relu')(l_2)
    l_4 = Dense(n_outputs, activation='sigmoid')(l_3)
    model = Model(inputs=x, outputs=l_4)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='figure.png',show_shapes=True)
    return model

# this LSTM uses the latent features from each time slice to make the final prediction
def TemporalWeightLstm(n_outputs,vocab_sz,emb_sz,embedding_matrix,n_timesteps):

    x = Input(shape=(n_timesteps,))
    # start creating a simpleLSTM model
    e = Embedding(vocab_sz, emb_sz, weights=[embedding_matrix], input_length=n_timesteps, trainable=True)(x)
    if DEVICE == 'CPU':
        l_1 = Bidirectional(GRU(100,activation='relu',return_sequences=True))(e)
    if DEVICE == 'GPU':
        l_1 = Bidirectional(CuDNNGRU(100, activation='relu',return_sequences=True))(e)
    l_2 = Dropout(0.2)(l_1)
    l_3 = Dense(100, activation='relu')(l_2)
    l_4 = Dense(n_outputs, activation='sigmoid')(l_3)
    model = Model(inputs=x, outputs=l_4)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='figure.png',show_shapes=True)
    return model

def MLP(n_outputs,vocab_sz,emb_sz,embedding_matrix,n_timesteps):

    x = Input(shape=(n_timesteps,))
    e = Embedding(vocab_sz, emb_sz, weights=[embedding_matrix], input_length=n_timesteps, trainable=True)(x)
    e = Flatten()(e)
    l_1 = Dense(100, activation='relu',kernel_initializer='normal')(e)
    l_2 = Dropout(0.2)(l_1)
    l_3 = Dense(100, activation='relu',kernel_initializer='normal')(l_2)
    l_4 = Dropout(0.2)(l_3)
    y = Dense(n_outputs, activation='sigmoid',kernel_initializer='normal')(l_4)
    model = Model(inputs=x, outputs=y)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model,to_file='SimpleMLP.png',show_shapes=True)
    return model

def StackedLSTMmodel(n_outputs,vocab_sz,emb_sz,embedding_matrix,n_timesteps):
    print('running stacked LSTM')
    hid_sz = 100
    x = Input(shape=(n_timesteps,))
    # start creating a simpleLSTM model
    e = Embedding(vocab_sz, emb_sz, weights=[embedding_matrix], input_length=n_timesteps, trainable=True)(x)
    if DEVICE == 'CPU':
        l_1 = Bidirectional(GRU(hid_sz, return_sequences=True, activation='relu'))(e)
        l_2 = Bidirectional(GRU(hid_sz, return_sequences=True, activation='relu'))(l_1)
    else:
        l_1 = Bidirectional(CuDNNGRU(hid_sz, return_sequences=True))(x)
        l_2 = Bidirectional(CuDNNGRU(hid_sz, return_sequences=True))(l_1)
    l_3 = TimeDistributed(Dense(100, activation='relu'))(l_2)
    l_4 = Flatten()(l_3)
    l_5 = Dense(n_outputs, activation='sigmoid')(l_4)
    model = Model(inputs=x, outputs=l_5)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='stackedLSTM_figure.png', show_shapes=True, show_layer_names=True)
    return model
