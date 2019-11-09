import pandas as pd
from TextModels import SimpleMLP
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from Utils import PurifyText
import numpy as np
from sklearn import preprocessing
'''
This program performs binary classification such as sentiments from a given text. 
1. Convert the word to Word2Vec embedings after some pre-processing.
2. For each document, average the Word2Vec to get a single D-dimensional(usually 100 or 300) vector per document. 
'''
if __name__  == '__main__':

    pth = '/home/vineeth/Documents/DataRepo/aclImdb/'
    fil = 'IMDB_50K_consolidated.tsv'
    # the name of the field for which you want to apply pre-processing
    req_fields = ['sentiment','review']
    cnt = 0
    verbose = 1
    batch_sz = 64
    folds = 10
    epochs = 100
    accuracies = []
    le = preprocessing.LabelEncoder()
    typ = 'word2vec averaging'
    df = pd.read_csv(pth+fil, sep='\t')[req_fields]
    # label encode the class label
    # df[req_fields[-1]] = le.fit_transform(df[req_fields[-1]])
    purify = PurifyText(typ)
    vals = purify.getText(df,req_fields[1])
    X,mx_doc_len = vals[0],vals[1]
    n_outputs = 1
    # check
    Y = df[req_fields[0]].values
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(df):
        cnt += 1
        weights_path = 'LearnedWeights/Model_'+str(cnt)+'.hdf5'
        X_train, X_test = X[train_index,], X[test_index,]
        Y_train, Y_test = Y[train_index,], Y[test_index,]
        # get validation data
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
        print('# train documents:{} \n # validation documents:{}'.format(len(X_train),len(X_valid)))
        # print('# unique words:{} \n embedding size of glove:{}'.format(uniq_wrds,emb_sz))
        callbcks_list = [ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_loss', mode='min',\
                          save_best_only=True),EarlyStopping(monitor='val_loss', mode='min',patience=5)]
        print('traning fold:{}'.format(cnt))
        model = SimpleMLP(mx_doc_len,n_outputs)
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_sz,callbacks=callbcks_list,\
              validation_data=(X_valid, Y_valid),verbose=verbose)

        # load the best parameters and test out the model
        model.load_weights(weights_path)
        print('loaded pre-trained weights for testing')
        _, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_sz, verbose=0)
        print(accuracy)
        accuracies.append(accuracy)
    print('average accuracy across {} folds: {}'.format(folds,np.average(accuracies)))
