import pandas as pd
from TextModels import SimpleLstm, EmbedMLP, StackedLSTMmodel
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from Utils import Embeddings
from Utils import PurifyText
import numpy as np
from sklearn import preprocessing
from os.path import expanduser

HOME = expanduser("~")


# MainA does not work
def MainA():
    pth = HOME + '/Documents/DataRepo/aclImdb/'
    fil = 'IMDB_50K_consolidated.csv'
    # the name of the field for which you want to apply pre-processing
    req_fields = ['sentiment', 'review']
    accuracies = []
    n_outputs = 1
    folds = 10
    cnt = 0
    epochs = 20
    batch_sz = 128
    verbose = 2
    typ = 'standard text processing'
    embed = Embeddings(typ)
    le = preprocessing.LabelEncoder()
    typ = 'standard text processing'
    df = pd.read_csv(pth + fil)[req_fields]
    # df = df.head(100)
    df[req_fields[0]] = le.fit_transform(df[req_fields[0]])
    purify = PurifyText(typ)
    vals = purify.getText(df, req_fields[1])
    X, mx_doc_len = vals[0], vals[1]
    Y = df[req_fields[0]].values
    print(mx_doc_len)
    if mx_doc_len > 500:
        mx_doc_len = 500
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)
    embed = Embeddings(typ)

    for train_index, test_index in kf.split(X):
        cnt += 1
        weights_path = 'LearnedWeights/Model_' + str(cnt) + '.hdf5'
        X_train, X_test = X[train_index,], X[test_index,]
        Y_train, Y_test = Y[train_index,], Y[test_index,]
        # get validation data
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
        padded_X_train, emb_matrix, emb_sz, vocab_sz = embed.getEmbeddings(X_train, mx_doc_len)
        padded_X_valid = embed.getEmbeddings(X_valid, mx_doc_len)[0]
        padded_X_test = embed.getEmbeddings(X_test, mx_doc_len)[0]
        uniq_wrds = emb_matrix.shape[0]
        print('# train documents:{} \n # validation documents:{}'.format(len(X_train), len(X_valid)))
        # print('# unique words:{} \n embedding size of glove:{}'.format(uniq_wrds,emb_sz))
        callbcks_list = [ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_loss', mode='min', \
                                         save_best_only=True), EarlyStopping(monitor='val_loss', mode='min', patience=5)]
        print('traning fold:{}'.format(cnt))
        model = EmbedMLP(n_outputs, vocab_sz, emb_sz, mx_doc_len, emb_matrix)
        model.fit(padded_X_train, Y_train, epochs=epochs, batch_size=batch_sz, callbacks=callbcks_list, \
                  validation_data=(padded_X_valid, Y_valid), verbose=verbose)

        # load the best parameters and test out the model
        model.load_weights(weights_path)
        print('loaded pre-trained weights for testing')
        _, accuracy = model.evaluate(padded_X_test, Y_test, batch_size=batch_sz, verbose=0)
        print(accuracy)
        accuracies.append(accuracy)
    print('average accuracy across {} folds: {}'.format(folds, np.average(accuracies)))


# this works
def MainB():
    pth = HOME + '/Documents/DataRepo/aclImdb/'
    fil = 'IMDB_50K_consolidated.csv'
    # the name of the field for which you want to apply pre-processing
    req_fields = ['sentiment', 'review']
    n_outputs = 1
    typ = 'standard text processing'
    embed = Embeddings(typ)
    le = preprocessing.LabelEncoder()
    typ = 'standard text processing'
    df = pd.read_csv(pth + fil)[req_fields]
    # df = df.head(100)
    df[req_fields[0]] = le.fit_transform(df[req_fields[0]])
    purify = PurifyText(typ)
    vals = purify.getText(df, req_fields[1])
    X, mx_doc_len = vals[0], vals[1]
    print(mx_doc_len)
    if mx_doc_len > 500:
        mx_doc_len = 500

    padded_docs, embedding_matrix, embed_sz, vocab_size = embed.getEmbeddings(X, mx_doc_len)
    # check
    Y = df[req_fields[0]].values
    X_train, X_test, Y_train, Y_test = train_test_split(padded_docs, Y, test_size=0.5)
    # model = EmbedMLP_test(n_outputs,vocab_size,32,mx_doc_len)
    model = EmbedMLP(n_outputs, vocab_size, 32, mx_doc_len)
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=128, verbose=2)
    # Final evaluation of the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    MainB()
