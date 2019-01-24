import numpy as np
import sys
sys.path.append('../')
from config import OUTPUT_DIR
import pandas as pd

class Recommender(object):
    
    def __init__(self, METHOD_NAME, DATA_NAME, MODEL_NAME=None, l0=None):
        self.DATA_NAME = DATA_NAME
        self.METHOD_NAME = METHOD_NAME
        
        if MODEL_NAME is None:
            params = [DATA_NAME, METHOD_NAME]
            self.MODEL_NAME = "_".join([str(p) for p in params])
        else:
            self.MODEL_NAME = MODEL_NAME+'_'+str(l0)
            
            
    def learner_config(self, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG, MAX_EPOCH, N_SAMPLE_PER_EPOCH):
        
        self.HIDDEN_DIM = HIDDEN_DIM
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.N_NEG = N_NEG
        self.MAX_EPOCH = MAX_EPOCH
        self.N_SAMPLE_PER_EPOCH = N_SAMPLE_PER_EPOCH
        
        DATA_NAME = self.DATA_NAME
        METHOD_NAME = self.METHOD_NAME
        HIDDEN_DIM = self.HIDDEN_DIM
        params = [DATA_NAME, METHOD_NAME, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG]
        self.MODEL_NAME = "_".join([str(p) for p in params])
        
        
    def assign_data(self, dataTrain, dataValidation, dataTest, n_user, n_item):
        self.dataTrain = dataTrain
        self.dataValidation = dataValidation
        self.dataTest = dataTest
        self.n_user = n_user
        self.n_item = n_item
        
        print('calculating item popularity and user-wise item frequency in training set ...')
        sys.stdout.flush()
        
        userTransMap = {}
        userItemFreq = {}
        itemFreq = np.zeros(n_item)
        for _k, (_u, _items) in enumerate(dataTrain):
            _transactions = userTransMap.get(_u, [])
            _transactions.append(_k)
            userTransMap[_u] = _transactions
            
            _itemFreq = userItemFreq.get(_u, {})
            for _i in _items:
                _itemFreq[_i] = _itemFreq.get(_i, 0) + 1
                itemFreq[_i] += 1
            userItemFreq[_u] = _itemFreq

        print('done!')
        
        self.userTransMap = userTransMap
        self.userItemFreq = userItemFreq
        self.itemFreq = itemFreq
        self.loyalty = {}

        sys.stdout.flush()

    
    def scoring(self, u, conditionItems):
        return None
    
    def predict(self, u, conditionItems):
        
        s = self.scoring(u, conditionItems)
        if u in self.loyalty:
            p_max = np.max(s)
            p = np.exp(s.copy() - p_max)
            p /= np.sum(p)
            
            s = p.copy()
            if u in self.loyalty:
                l_u = self.loyalty[u]
                for i in l_u:
                    q, l = l_u[i]
                    s[i] = q*l + p[i]*(1-l)
        return s             


    def update_loyalty(self, l0):
        eps = 1e-10

        loyalty = {}
        dataTrain = self.dataTrain
        userTransMap = self.userTransMap

        print("udpating loyalty ... ")
        sys.stdout.flush()        
        
        print("current user: ", end="")
        countU = 0
        for _u in userTransMap:
            if countU % 5000 == 0:
                print(countU, end=", ")
                sys.stdout.flush()
            countU += 1
            
            s = self.scoring(_u, None)
            p_max = np.max(s)
            p = np.exp(s - p_max)
            p /= np.sum(p)
                
            l_u = {}
            count = 0
            for _t in userTransMap[_u]:
                _items = dataTrain[_t][1]
                _negs = set(l_u.keys()) - set(_items)
                n = len(_items)
                for _i in _items:
                    q, l = 0, l0
                    if _i in l_u:
                        q, l = l_u[_i]
                        l = (q*l + eps)/(q*l + p[_i]*(1-l) + eps)
                    q = (q*count + 1)/(count + n)
                    l_u[_i] = [q,l]
                for _i in _negs:
                    q, l = l_u[_i]
                    l = ((1-q)*l + eps)/((1-q)*l + (1-p[_i])*(1-l) + eps)
                    q = (q*count)/(count + n)
                    l_u[_i] = [q,l]
                count += n
            loyalty[_u] = l_u
        self.loyalty = loyalty
        print("done!")
        sys.stdout.flush()
        
    
    def evaluate_user(self, u, testItems, maskItems, conditionItems):
        s_u = self.predict(u, conditionItems)
        n_item = self.n_item
        neg_index = np.ones(n_item)
        neg_index[maskItems] = 0
        s_target = s_u[testItems]
        s_neg = s_u[neg_index>0]
        n_neg = len(s_neg)
        wrong = (s_target.reshape(1, len(s_target)) <= s_neg.reshape(n_neg,1)).sum(axis=0)
        auc = (n_neg - wrong) / n_neg
        ndcg = 1.0/np.log2(2 + wrong)
        return auc, ndcg
        
    def evaluate(self, dump=True):
        userItemFreq = self.userItemFreq
        dataValidation = self.dataValidation
        dataTest = self.dataTest
        
        print("evaluating validation data ... ")
        metricAll = []
        metricWarm = []
        metricCold = []
        count = 0
        print("current progress: ", end="")
        sys.stdout.flush()
        for u, items in dataValidation:
            _auc, _ndcg = self.evaluate_user(u, items, items, None)
            metricAll.append([_auc.mean(), _ndcg.mean()])
            if count % 10000 == 0:
                print(count, end=", ")
                sys.stdout.flush()
            count += 1
            if u in userItemFreq:
                isWarm = np.array([int(items[ki] in userItemFreq[u]) for ki in range(len(items))])
                if np.sum(isWarm)>0:
                    metricWarm.append([_auc[isWarm>0].mean(), _ndcg[isWarm>0].mean()])
                if np.sum(isWarm)<len(isWarm):
                    metricCold.append([_auc[isWarm<1].mean(), _ndcg[isWarm<1].mean()])
            else:
                metricCold.append([_auc.mean(), _ndcg.mean()])
        metricValidation = np.array([np.array(metricAll).mean(axis=0), 
                                      np.array(metricWarm).mean(axis=0),
                                      np.array(metricCold).mean(axis=0)])
        print("complete!")
        sys.stdout.flush()
        
        print("evaluating test data ... ")
        metricAll = []
        metricWarm = []
        metricCold = []
        count = 0
        print("current progress: ", end="")
        sys.stdout.flush()
        for u, items in dataTest:
            items = np.array(items)
            _auc, _ndcg = self.evaluate_user(u, items, items, None)
            metricAll.append([_auc.mean(), _ndcg.mean()])
            if count % 10000 == 0:
                print(count, end=", ")
                sys.stdout.flush()
            count += 1
            if u in userItemFreq:
                isWarm = np.array([int(items[ki] in userItemFreq[u]) for ki in range(len(items))])
                if np.sum(isWarm)>0:
                    metricWarm.append([_auc[isWarm>0].mean(), _ndcg[isWarm>0].mean()])
                if np.sum(isWarm)<len(isWarm):
                    metricCold.append([_auc[isWarm<1].mean(), _ndcg[isWarm<1].mean()])
            else:
                metricCold.append([_auc.mean(), _ndcg.mean()])
        metricTest = np.array([np.array(metricAll).mean(axis=0), 
                                      np.array(metricWarm).mean(axis=0),
                                      np.array(metricCold).mean(axis=0)])        
        print("complete!")
        sys.stdout.flush()
        print("done!")
        dfVali = pd.DataFrame(metricValidation, columns=['auc','ndcg'], index=['all','warm','cold'])
        dfTest = pd.DataFrame(metricTest, columns=['auc','ndcg'], index=['all','warm','cold'])
        if dump:
            
            dfVali.to_csv(OUTPUT_DIR+self.MODEL_NAME+".recommendation.results.validation.csv")
            dfTest.to_csv(OUTPUT_DIR+self.MODEL_NAME+".recommendation.results.test.csv")
        
        print('validation results')
        print(dfVali.round(3))
        print('test results')
        print(dfTest.round(3))
        sys.stdout.flush()
        
        return dfVali, dfTest
        
        

