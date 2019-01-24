import tensorflow as tf
from embedding.learner import Model
from embedding.sampler import Sampler
import sys

class triple2vec(Model):
    
    def __init__(self, DATA_NAME, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG, MAX_EPOCH=500, N_SAMPLE_PER_EPOCH=None):
        super().__init__('triple2vec', DATA_NAME, HIDDEN_DIM, LEARNING_RATE, BATCH_SIZE, N_NEG, MAX_EPOCH, N_SAMPLE_PER_EPOCH)
        
    def assign(self, dataTrain, n_user, n_item, N_SAMPLE, dump=True):
        mySampler = Sampler(dataTrain, self.DATA_NAME)
        
        trainSamples = mySampler.sample_triples(N_SAMPLE, dump=dump)
        super().assign_data(trainSamples, n_user, n_item)
        
    def assign_from_file(self, n_user, n_item):
        mySampler = Sampler(None, self.DATA_NAME)
        trainSamples = mySampler.load_triples_from_file()
        super().assign_data(trainSamples, n_user, n_item)
        
    def model_constructor(self, opt='sgd'):
        
        n_user = self.n_user
        n_item = self.n_item
        HIDDEN_DIM = self.HIDDEN_DIM
        LEARNING_RATE = self.LEARNING_RATE
        N_NEG = self.N_NEG
            
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])
    
        user_emb = tf.get_variable("user_emb", [n_user, HIDDEN_DIM], 
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01))
        item_emb1 = tf.get_variable("item_emb1", [n_item, HIDDEN_DIM], 
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01))
        item_emb2 = tf.get_variable("item_emb2", [n_item, HIDDEN_DIM], 
                                     initializer=tf.random_uniform_initializer(-0.01, 0.01))        
        b_item = tf.get_variable("item_bias", [n_item, 1], 
                                initializer=tf.constant_initializer(0))
        b_user = tf.get_variable("user_bias", [n_user, 1], 
                                initializer=tf.constant_initializer(0))
        
        i_emb = tf.nn.embedding_lookup(item_emb1, i)
        j_emb = tf.nn.embedding_lookup(item_emb2, j)
        u_emb = tf.nn.embedding_lookup(user_emb, u)
        
        input_emb_i = j_emb + u_emb
        loss_i = tf.reduce_mean(tf.nn.nce_loss(weights=item_emb1, biases=b_item[:,0],
                         labels=tf.reshape(i, (tf.shape(i)[0], 1)), inputs=input_emb_i, 
                         num_sampled=N_NEG, num_classes=n_item))
        input_emb_j = i_emb + u_emb
        loss_j = tf.reduce_mean(tf.nn.nce_loss(weights=item_emb2, biases=b_item[:,0],
                         labels=tf.reshape(j, (tf.shape(j)[0], 1)), inputs=input_emb_j, 
                         num_sampled=N_NEG, num_classes=n_item))
        input_emb_u = i_emb + j_emb
        loss_u = tf.reduce_mean(tf.nn.nce_loss(weights=user_emb, biases=b_user[:,0],
                         labels=tf.reshape(u, (tf.shape(u)[0], 1)), inputs=input_emb_u, 
                         num_sampled=N_NEG, num_classes=n_user))     
        trainloss = tf.reduce_mean([loss_i, loss_j, loss_u])
        
        if opt == 'sgd':
            myOpt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        elif opt == 'adaGrad':
            myOpt = tf.train.AdagradOptimizer(LEARNING_RATE)
        elif opt == 'adam':
            myOpt = tf.train.AdamOptimizer(LEARNING_RATE)
        elif opt == 'lazyAdam':
            myOpt = tf.contrib.opt.LazyAdamOptimizer(LEARNING_RATE)
        elif opt == 'momentum':
            myOpt = tf.train.MomentumOptimizer(LEARNING_RATE, 0.9)
        else:
            print('optimizer is not recognized, use SGD instead.')
            sys.stdout.flush()
            myOpt = tf.train.GradientDescentOptimizer(LEARNING_RATE)
        optimizer = myOpt.minimize(trainloss)

        paramDict = {'item_emb1': item_emb1, 'item_emb2': item_emb2, 'user_emb': user_emb, 'item_bias': b_item, 'user_bias': b_user}
        return [u, i, j], [trainloss], [optimizer], paramDict 
    

