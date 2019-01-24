import pandas as pd
import argparse
import os
dirname, _ = os.path.split(os.path.abspath('__file__'))
dirs = [d for d in os.listdir(dirname) if os.path.isdir(d)]
import sys
for d in dirs:
    sys.path.append(d)
    
from config import DATA_DIR
from embedding.triple2vec import triple2vec
from recommendation.recommender import popRec, popUserRec, triple2vecRec

def load_data(DATA_NAME):
    
    print('loading', DATA_NAME, 'data ...')
    myTrans = pd.read_csv(DATA_DIR + DATA_NAME + ".data.csv", encoding = 'latin1')
    myTrans['PID'] = myTrans['PID'].apply(lambda x : list(set(eval(x))))
    myItem = pd.read_csv(DATA_DIR + DATA_NAME + ".meta.csv", encoding = 'latin1')
    n_item = len(myItem)
    n_user = myTrans['UID'].max() + 1
    print('done!')
    print('interactions about', n_item, 'products and', n_user, 'users are loaded')
    return myTrans, myItem, n_item, n_user


def run_embedding(DATA_NAME, METHOD_NAME, dim, lr, batch_size, n_neg):

    myTrans, myItem, n_item, n_user = load_data(DATA_NAME)
    dataTrain = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'train'].values
    
    embeddingDict = None
    if METHOD_NAME == 'triple2vec':
        myModel = triple2vec(DATA_NAME=DATA_NAME, 
                             HIDDEN_DIM=dim, LEARNING_RATE=lr, BATCH_SIZE=batch_size, 
                             N_NEG=n_neg, MAX_EPOCH=500, N_SAMPLE_PER_EPOCH=None)
        myModel.assign(dataTrain, n_user, n_item, N_SAMPLE=5000000, dump=True)
        #myModel.assign_from_file(n_user, n_item)
        myModel.train(opt='momentum')
        embeddingDict = myModel.extract_emebdding(True)
    return embeddingDict


def run_recommendation(data_name, method_name, dim=None, lr=None, batch_size=None, n_neg=None, l0=None):

    myTrans, myItem, n_item, n_user = load_data(data_name)
    dataTrain = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'train'].values
    dataValidation = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'validation'].values
    dataTest = myTrans[['UID', 'PID']].loc[myTrans['flag'] == 'test'].values
    
    if method_name == 'triple2vec':
        params = [data_name, 'triple2vec', dim, lr, batch_size, n_neg]
        model_name = "_".join([str(p) for p in params])
        myRec = triple2vecRec(data_name, model_name, l0=l0)
        myRec.assign_data(dataTrain, dataValidation, dataTest, n_user, n_item)
        myRec.assign_embeddings()
        resVali, resTest = myRec.evaluate(dump=True)
    elif method_name == 'popRec':
        myRec = popRec(data_name)
        myRec.assign_data(dataTrain, dataValidation, dataTest, n_user, n_item)
        resVali, resTest = myRec.evaluate(dump=True)
    elif method_name == 'popUserRec':
        myRec = popUserRec(data_name)
        myRec.assign_data(dataTrain, dataValidation, dataTest, n_user, n_item)
        resVali, resTest = myRec.evaluate(dump=True) 
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='instacart',
                        help="")
    parser.add_argument('--mode', default='embedding',
                        help="")
    parser.add_argument('--method_name', default='triple2vec')
    parser.add_argument('--dim', default=32, type=int)
    parser.add_argument('--lr', default=1.0, type=float)
    parser.add_argument('--batch_size', default=1000, type=int)
    parser.add_argument('--n_neg', default=5, type=int)
    parser.add_argument('--l0', default=-1, type=float)
    
    args = parser.parse_args()
    data_name, mode, method_name = args.data_name, args.mode, args.method_name
    if mode == 'embedding':
        dim, lr, batch_size, n_neg = args.dim, args.lr, args.batch_size, args.n_neg
        run_embedding(data_name, method_name, dim, lr, batch_size, n_neg)
    elif mode == 'recommendation':
        if (method_name == 'popRec') or (method_name == 'popUserRec'):
            run_recommendation(data_name, method_name)
        elif method_name == 'triple2vec':
            dim, lr, batch_size, n_neg, l0 = args.dim, args.lr, args.batch_size, args.n_neg, args.l0
            if l0 < 0:
                l0 = None
            run_recommendation(data_name, method_name, dim, lr, batch_size, n_neg, l0)


