import numpy as np
from recommendation.model import Recommender
from config import PARAM_DIR

class popRec(Recommender):

    def __init__(self, DATA_NAME):
        super().__init__('popRec', DATA_NAME)
        
    def scoring(self, u, conditionItems):
        np.random.seed(0)
        s = np.random.rand(self.n_item)*1e-2
        s += self.itemFreq.copy()
        return s
    

class popUserRec(Recommender):

    def __init__(self, DATA_NAME):
        super().__init__('popUserRec', DATA_NAME)
        
    def scoring(self, u, conditionItems):
        np.random.seed(0)
        s = np.random.rand(self.n_item)*1e-2
        if u in self.userItemFreq:
            for (i, v) in self.userItemFreq[u].items():
                s[i] += v
        return s

    

class triple2vecRec(Recommender):
    
    def __init__(self, DATA_NAME, MODEL_NAME, l0, ensemble=False):
        super().__init__('triple2vec', DATA_NAME, MODEL_NAME, l0)
        self.ensemble = ensemble
        self.l0 = l0
        self.FILE_NAME = MODEL_NAME
    
    def assign_embeddings(self, params=None):
        if params is None:
            try:
                item_emb1 = np.genfromtxt(PARAM_DIR + self.FILE_NAME + ".item_emb1.csv", delimiter=", ")
                item_emb2 = np.genfromtxt(PARAM_DIR + self.FILE_NAME + ".item_emb2.csv", delimiter=", ")
                item_emb = (item_emb1 + item_emb2)/2.0
                user_emb = np.genfromtxt(PARAM_DIR + self.FILE_NAME + ".user_emb.csv", delimiter=", ")
                item_bias = np.genfromtxt(PARAM_DIR + self.FILE_NAME + ".item_bias.csv", delimiter=", ")
            except:
                print('user/item embeddings from '+self.FILE_NAME+' are not saved under', PARAM_DIR)
        else:
            item_emb1, item_emb2, user_emb, item_bias = params
            item_emb = (item_emb1 + item_emb2)/2.0

        self.item_emb1 = item_emb1
        self.item_emb2 = item_emb2            
        self.item_emb = item_emb
        self.user_emb = user_emb
        self.item_bias = item_bias
        
        if self.l0 is not None:
            self.update_loyalty(self.l0)
            
            
        
    def scoring(self, u, conditionItems):
        
        s = np.dot(self.item_emb, self.user_emb[u,:]) + self.item_bias.copy()
        
        if self.ensemble and u in self.userItemFreq:
            u_emb1 = np.zeros(self.user_emb.shape[1])
            u_emb2 = np.zeros(self.user_emb.shape[1])
            vSum = 0
            for _i, _v in self.userItemFreq[u].items():
                u_emb1 += self.item_emb1[_i, :]*_v
                u_emb2 += self.item_emb2[_i, :]*_v
                vSum += _v
            u_emb1 /= vSum
            u_emb2 /= vSum
            s += (np.dot(self.item_emb2, u_emb1) + np.dot(self.item_emb1, u_emb2))/2.0
            
        return s

