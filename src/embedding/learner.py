import numpy as np
import tensorflow as tf
import sys
sys.path.append('../')
from config import MODEL_DIR, PARAM_DIR


class Model(object):

    def __init__(self, METHOD_NAME, DATA_NAME, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG, MAX_EPOCH, N_SAMPLE_PER_EPOCH):
        self.DATA_NAME = DATA_NAME
        self.HIDDEN_DIM = HIDDEN_DIM
        self.LEARNING_RATE = LEARNING_RATE
        self.BATCH_SIZE = BATCH_SIZE
        self.N_NEG = N_NEG
        self.MAX_EPOCH = MAX_EPOCH
        self.N_SAMPLE_PER_EPOCH = N_SAMPLE_PER_EPOCH
        
        params = [DATA_NAME, METHOD_NAME, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG]
        self.MODEL_NAME = "_".join([str(p) for p in params])
        
        
    def assign_data(self, trainSamples, n_user, n_item):
        self.trainSamples = trainSamples
        self.n_user = n_user
        self.n_item = n_item
        

    def next_batch(self):
        trainSamples = self.trainSamples
        BATCH_SIZE = self.BATCH_SIZE
        if self.N_SAMPLE_PER_EPOCH is None:
            N_SAMPLE_PER_EPOCH = trainSamples.shape[0]
        else:
            N_SAMPLE_PER_EPOCH = min(trainSamples.shape[0], self.N_SAMPLE_PER_EPOCH)
        
        N_BATCH = N_SAMPLE_PER_EPOCH // BATCH_SIZE

        index_selected = np.random.permutation(self.trainSamples.shape[0])[:N_SAMPLE_PER_EPOCH]
        
        for i in range(0, N_BATCH*BATCH_SIZE, BATCH_SIZE):
            current_index = index_selected[i:(i+BATCH_SIZE)]
            xu1 = trainSamples[current_index, 0]
            xi1 = trainSamples[current_index, 1]
            xj1 = trainSamples[current_index, 2]
            yield xu1, xi1, xj1


    def train(self, opt='sgd'):   
        MODEL_NAME = self.MODEL_NAME
        MAX_EPOCH = self.MAX_EPOCH
        max_noprogress = 5
        
        print("start training "+MODEL_NAME+" ...")
        sys.stdout.flush()
        config = tf.ConfigProto()
        with tf.Graph().as_default(), tf.Session(config=config) as session:
            variables, losses, optimizers, paramDict = self.model_constructor(opt)
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()

            _loss_train_min = 1e10
            n_noprogress = 0
            
            for epoch in range(1, MAX_EPOCH):
                _count, _count_sample = 0, 0
                _loss_train = [0 for _ in range(len(losses))]

                print("epoch: ", epoch)
                print("=== current batch: ", end="")
                for _vars in self.next_batch():

                    feed = dict(zip(variables, _vars))

                    _loss_batch, _ = session.run([losses, optimizers],
                                                 feed_dict=feed)
                    for _i, _l in enumerate(_loss_batch):
                        _loss_train[_i] += _l

                    _count += 1.0
                    _count_sample += _vars[0].shape[0]
                    if _count % 500 == 0:
                        print(int(_count), end=", ")
                        sys.stdout.flush()
                print("complete!")
                sys.stdout.flush()

                for _i in range(len(_loss_train)):
                    _loss_train[_i] /= _count

                if _loss_train[0] < _loss_train_min:
                    _loss_train_min = _loss_train[0]
                    n_noprogress = 0
                    saver.save(session, MODEL_DIR +
                               self.MODEL_NAME + ".model.ckpt")
                else:
                    n_noprogress += 1

                print("=== training: primary loss: {:.4f}, min loss: {:.4f}".format(
                    _loss_train[0], _loss_train_min), end=";  ")
                for _i, _l in enumerate(_loss_train[1:]):
                    print(" aux_loss" + str(_i) +
                          ": {:.4f}".format(_l), end="")
                print("")
                sys.stdout.flush()

                print("=== #no progress: ", n_noprogress)
                sys.stdout.flush()

                if n_noprogress >= max_noprogress:
                    break
            saver.restore(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")
        print("done!")
        sys.stdout.flush()
        
        
    def extract_emebdding(self, dump=False):
        print("Restoring the model graph and extracting embeddings ...")
        sys.stdout.flush()
        config = tf.ConfigProto()
        res = {}
        with tf.Graph().as_default(), tf.Session(config=config) as session:
            _, _, _, paramDict = self.model_constructor()
            # params will be a {name: variable} dictionary
            session.run(tf.global_variables_initializer())
            saver = tf.train.Saver()        
            saver.restore(session, MODEL_DIR + self.MODEL_NAME + ".model.ckpt")

            for name, param in paramDict.items():
                _param = session.run(param)
                res[name] = _param
                if dump:
                    np.savetxt(PARAM_DIR + self.MODEL_NAME + "."+name+".csv", _param, delimiter=", ")
                    print(name, " is dumped in ", PARAM_DIR + self.MODEL_NAME + "."+name+".csv")
                sys.stdout.flush()
        print("done!")
        sys.stdout.flush()
        
        return res
        
        

