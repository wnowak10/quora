import pandas as pd
from create_features import *
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np


x_train = df_train.drop(['question1', 'question2','q1words','q2words','id','qid1','qid2','is_duplicate'], axis=1)
x_test = df_test.drop(['question1', 'question2','q1words','q2words','test_id',], axis=1)
# x_train = df_train.iloc[:, [6, 7, 10, 11, 12, 13, 14, 15]]
# x_test = df_test.iloc[:, [3, 4, 7, 8, 9, 10, 11, 12]]
y_train = df_train['is_duplicate'].values

print('x_train and y_train chopped and ready')
# rescale (copy anokas)

pos_train = x_train[y_train == 1]
neg_train = x_train[y_train == 0]

# Now we oversample the negative class
# There is likely a much more elegant way to do this...
p = 0.165
scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
while scale > 1:
    neg_train = pd.concat([neg_train, neg_train])
    scale -= 1
neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
print(len(pos_train) / (len(pos_train) + len(neg_train)))

x_train = pd.concat([pos_train, neg_train])
y_train = (np.zeros(len(pos_train)) + 1).tolist() + np.zeros(len(neg_train)).tolist()
del pos_train, neg_train


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2)

print('data rescaled and ready to train')

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]

bst = xgb.train(params, d_train, 5, watchlist, early_stopping_rounds=50, verbose_eval=10)


d_test = xgb.DMatrix(x_test)
p_test = bst.predict(d_test)

sub = pd.DataFrame()
sub['test_id'] = df_test['test_id']
sub['is_duplicate'] = p_test
sub.to_csv('simple_xgb.csv', index=False)

