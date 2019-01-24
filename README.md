# Grocery

This repo includes python/tensorflow implementations of several algorithms proposed for modeling grocery shopping behavior.

These alogrithms have been implemented

 - the product representation learning model -- **triple2vec**
 - the incremental module -- **adaLoyal** for next-basket 
recommendations

both proposed in

**Mengting Wan, Di Wang, Jie Liu, Paul Bennett, Julian McAuley, "Representing and Recommending Shopping Baskets with Complementarity, Compatibility, and Loyalty", in CIKM'18** [[bibtex]](https://dblp.uni-trier.de/rec/bibtex/conf/cikm/WanWLBM18)

Algorithms proposed in

**Mengting Wan, Di Wang, Matt Goldman, Matt Taddy, Justin Rao, Jie Liu, Dimitrios Lymberopoulos, Julian McAuley, "Modeling Consumer Preferences and Price Sensitivities from Large-Scale Grocery Shopping Transaction Logs", in WWW'17** [[bibtex]](https://dblp.uni-trier.de/rec/bibtex/conf/www/WanWGTRLLM17)

will be added in the future.

If you would like to extend or compare with our algorithms, or use our source code, please consider citing the above two papers.

## Quick Start

Requirement:

 - Python 3.6+ (older version has not been tested)
 - Tensorflow 1.6.0+ (older version has not been tested)

### Quick start with a subset of *Instacart* for triple2vec and adaLoyal

 - Please first download the complete dataset from [here](https://www.instacart.com/datasets/grocery-shopping-2017) and release the files under `./data/`. This is a relatively large dataset which includes more than 3 million orders. We could start with a small subset of users to test the algorithms instantly.

 - **Preprocess the dataset**
 
 > python ./src/parser.py --data\_name instacart --thr\_item 10 --thr\_user --subset\_user 0.1
 
 This will randomly sample transactions associated with 10% users and filter out products with <10 transactions. Please consider adjusting these thresholds if you plan to run the algorithms on the complete dataset.
 
 The processed files will be saved as 

  + `./data/instacart.data.csv`: csv file which can be read by `pandas` and must include the following columns: `UID`(integers used to represent user IDs), `PID`(a list of integers to represent product IDs in the current transaction), `flag`(train, validation or test). Each row represents each transaction/basket record.
 
  + `./data/instacart.meta.csv`: csv file which can be read by `pandas`, including meta-data of products.

    **Note:** In order to run *adaLoyal*, transactions in this file **need to be sorted in chronological order**.
    
    **Note:** In order to run *triple2vec*, product IDs **need to be sorted based on their popularities** (i.e., PID=0 represents the most popular product). This will boost the negative sampling process in the noise contrastive estimation loss functions applied in representation learning algorithms.

- **Run _triple2vec_**

> python ./src/main.py --data\_name instacart --mode embedding --method\_name triple2vec --dim 32 --lr 1.0 --batch\_size 1000 --n\_neg 5

This will first generate training samples and cache it under `./output/sample/` (optional). Then product and user embeddings will be dumped under `./output/param/`.

- **Run personalized recommendation using item/user generated from _triple2vec_**

> python ./src/main.py --data\_name instacart --mode recommendation --method\_name triple2vec --dim 32 --lr 1.0 --batch\_size 1000 --n\_neg 5

- **Run personalized recommendation using item/user generated from _triple2vec_ and apply _adaLoyal_**

> python ./src/main.py --data\_name instacart --mode recommendation --method\_name triple2vec --dim 32 --lr 1.0 --batch\_size 1000 --n\_neg 5 --l0 0.8

where the initial loyalty is set as `l0=0.8`.

All results will be saved under `./output/result/`.

We can also test some simple baselines on this dataset

- rank products based on their overall popularities in the training set

> python ./src/main.py --data\_name instacart --mode recommendation --method\_name popRec 

- rank products based on user-wise item purchase frequency

> python ./src/main.py --data\_name instacart --mode recommendation --method\_name popRec 

### How to apply _adaLoyal_ on top of user/product representations from other models?

Ad-hoc needs can be added in this module `./src/recommendation/recommender.py`.


## TO-DO
 - Implement within-basket recommendation
 - Complete documentation
 - Implement price sensitivities

