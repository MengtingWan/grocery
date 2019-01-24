import numpy as np
import pandas as pd
import time
from collections import Counter
import argparse
from config import DATA_DIR

def parse_instacart(thr_i = 10, thr_u = 0, subset_user = 0.1):
    PATH_IN = DATA_DIR
    PATH_OUT = DATA_DIR + "instacart"
    start = time.time()
    print("loading data ...")
    
    transItem_prior = pd.read_csv(PATH_IN+"order_products__prior.csv")
    transItem_train = pd.read_csv(PATH_IN+"order_products__train.csv")
    myTransItem = pd.concat([transItem_prior, transItem_train])[['order_id', 'product_id']]
    myTransItem.columns = ['TID', 'PID']
    del transItem_prior, transItem_train
    myTrans = pd.read_csv(PATH_IN+"orders.csv")[['order_id', 'user_id', 'order_number', 'eval_set']]
    myTrans.columns = ['TID', 'UID', 'TS', 'flag']
    tmp = myTrans['UID'].value_counts()
    user_count = tmp.reset_index()
    user_count.columns = ['UID', 'count']
    user_count['select'] = (np.random.rand(user_count.shape[0]) < subset_user)
    myTrans = pd.merge(myTrans, user_count.loc[(user_count['select']) & (user_count['count']>=thr_u)], on = 'UID')
    
    myItem = pd.read_csv(PATH_IN+"products.csv")
    myItem.columns = ['PID', 'description', 'categoryId', 'departmentId']
    
    tmp = pd.read_csv(PATH_IN+"departments.csv")
    tmp.columns = ['departmentId', 'department']
    myItem = pd.merge(myItem, tmp, on='departmentId')
    
    tmp = pd.read_csv(PATH_IN+"aisles.csv")
    tmp.columns = ['categoryId', 'category']
    myItem = pd.merge(myItem, tmp, on='categoryId')
    
    
    print("done!")
    print("loading took {0:.1f} sec".format(time.time() - start))
    
    print("processing data ...")
    start = time.time()
    myTransItem = pd.merge(myTransItem, myTrans[['TID']], on = 'TID')
    tmp = myTransItem['PID'].value_counts()
    
    item_count = tmp.reset_index()
    item_count.columns = ['PID', 'count']
    
    count_i = np.array(list(Counter(item_count["count"].values).items()))
    count_i = count_i[np.argsort(count_i[:,0]), :]
    
    item_descr = pd.merge(myItem, item_count.loc[item_count["count"]>=thr_i], on = 'PID').sort_values(['count'], ascending=False)
    
    myTransItem = pd.merge(myTransItem, item_descr[['PID']], on = 'PID')
    item_list = item_descr['PID'].values
    item_dict = dict(zip(item_list, np.arange(len(item_list))))
    
    myData = pd.merge(myTrans, myTransItem, on = 'TID')
    myData.loc[myData['flag']=='train', 'flag'] = 'test'
    myData.loc[myData['flag']=='prior', 'flag'] = 'train'
    user_list = np.array(list(set(myData['UID'].values)))
    user_dict = dict(zip(user_list, np.arange(len(user_list))))
    myData = myData.groupby(['flag', 'UID', 'TS'])['PID'].apply(lambda x: [item_dict[k] for k in x])
    myData = myData.reset_index()
    myData['UID'] = myData['UID'].apply(lambda x : user_dict[x])
    myData = myData.sort_values(['TS'])
    
    tmp_u = myData.loc[myData['flag']=='train'].groupby('UID')
    tmp = []
    for (k, d) in tmp_u:
        if len(d) > 1:
            tmp.append(d.index[-1])
    np.random.seed(0)
    ind_validation = tmp
    myData.loc[ind_validation, "flag"] = "validation"
    
    del myTrans, myTransItem, myItem

    item_descr['PID'] = item_descr['PID'].apply(lambda x: [item_dict[k] for k in x])
    item_descr.to_csv(PATH_OUT+".meta.csv")
    myData.to_csv(PATH_OUT+".data.csv")

    print("done!")
    print("processing took {0:.1f} sec".format(time.time() - start))
    
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', default='instacart')
    parser.add_argument('--thr_item', default=10, type=int)
    parser.add_argument('--thr_user', default=0, type=int)
    parser.add_argument('--subset_user', default=0.1, type=float)
    args = parser.parse_args()
    
    if args.data_name == 'instacart':
        parse_instacart(thr_i = args.thr_item, thr_u = args.thr_user, subset_user = args.subset_user)