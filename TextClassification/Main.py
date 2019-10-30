import pandas as pd
from TextModels import SimpleLstm, MLP, StackedLSTMmodel
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold, train_test_split
from Utils import Embeddings
from Utils import PurifyText
import numpy as np

if __name__  == '__main__':

    pth = '<pathToFile>'
    fil = '<file>.csv'
    # the name of the field for which you want to apply pre-processing
    field = '<Name of the text field>'
    req_fields = ['<Name of the text field>','label']
    cnt = 0
    verbose = 0
    batch_sz = 64
    folds = 10
    epochs = 100
    accuracies = []
    typ = 'glove'
    df = pd.read_csv(pth+fil)[req_fields]
    vals = PurifyText(df,field)
    df,mx_doc_len = vals[0],vals[1]
    n_outputs = 1
    X, Y = df['<Name of the text field>'].values, df['label'].values
    kf = KFold(n_splits=folds, shuffle=True)
    kf.get_n_splits(X)
    embed = Embeddings(typ)
    for train_index, test_index in kf.split(df):
        cnt += 1
        weights_path = 'LearnedWeights/Model_'+str(cnt)+'.hdf5'
        X_train, X_test = X[train_index,], X[test_index,]
        Y_train, Y_test = Y[train_index,], Y[test_index,]
        # get validation data
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1, random_state=42)
        padded_X_train,emb_matrix,emb_sz,vocab_sz = embed.getEmbeddings(X_train,mx_doc_len)
        padded_X_valid = embed.getEmbeddings(X_valid,mx_doc_len)[0]
        padded_X_test = embed.getEmbeddings(X_test,mx_doc_len)[0]
        uniq_wrds = emb_matrix.shape[0]
        print('# train documents:{} \n # validation documents:{}'.format(len(X_train),len(X_valid)))
        # print('# unique words:{} \n embedding size of glove:{}'.format(uniq_wrds,emb_sz))
        callbcks_list = [ModelCheckpoint(filepath=weights_path, verbose=1, monitor='val_loss', mode='min',\
                          save_best_only=True),EarlyStopping(monitor='val_loss', mode='min',patience=5)]
        print('traning fold:{}'.format(cnt))
        model = StackedLSTMmodel(n_outputs,vocab_sz,emb_sz,emb_matrix,mx_doc_len)
        model.fit(padded_X_train, Y_train, epochs=epochs, batch_size=batch_sz,callbacks=callbcks_list,\
              validation_data=(padded_X_valid, Y_valid),verbose=verbose)

        # load the best parameters and test out the model
        model.load_weights(weights_path)
        print('loaded pre-trained weights for testing')
        _, accuracy = model.evaluate(padded_X_test, Y_test, batch_size=batch_sz, verbose=0)
        print(accuracy)
        accuracies.append(accuracy)
    print('average accuracy across {} folds: {}'.format(folds,np.average(accuracies)))
