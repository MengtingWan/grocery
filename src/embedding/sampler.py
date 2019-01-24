import numpy as np
#import multiprocessing as mp
import sys
from config import SAMPLE_DIR


class Sampler(object):
    
    def __init__(self, dataTrain, DATA_NAME):
        self.dataTrain = dataTrain
        self.DATA_ANME = DATA_NAME
        print("successfully initialized!")
        sys.stdout.flush()


    def sample_triples(self, N_SAMPLE, dump=False):
        print("preparing training triples ... ")
        sys.stdout.flush()
        print("current progress for", N_SAMPLE, "samples: ", end=" ")
        sys.stdout.flush()        
        n_interactions = self.dataTrain.shape[0]
        sampled_index = np.random.choice(n_interactions, size=N_SAMPLE)
        res = []
        for _k in range(N_SAMPLE):
            if _k % 100000 == 0:
                print(_k, end=", ")
                sys.stdout.flush()       
            _u, _items = self.dataTrain[sampled_index[_k]]
            _i, _j = np.random.choice(_items, size=2)
            res.append([_u, _i, _j])
        print("done!")
        
        res = np.array(res)
        if dump:
            np.savetxt(SAMPLE_DIR + self.DATA_ANME+".triples.csv", res, delimiter=", ")
        return res
    
    def load_triples_from_file(self):
        res = np.genfromtxt(SAMPLE_DIR + self.DATA_ANME+".triples.csv", delimiter=", ")
        return res
    
        

    