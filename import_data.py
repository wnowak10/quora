
# coding: utf-8

# Building off of
# [Anokas](https://www.kaggle.com/anokas/quora-question-pairs/data-analysis-xgboost-starter-0-35460-lb/run/1016915),
# trying to use a variety of methods and models to solve!

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

print('reading training set')
# df_train = pd.read_csv('df_train.csv',
# 						header=0,
# 						sep=',',
# 						quotechar='"')

# df_test = pd.read_csv('df_test.csv',
# 						header=0,
# 						sep=',',
# 						quotechar='"')


# read test and train from kaggle
# full_df_train = pd.read_csv('../input/train.csv')
# full_df_test = pd.read_csv('../input/train.csv')

np.random.seed(1234)
df_train = df_train.sample(1000)
df_test = df_test.sample(1000)


# print('head of df_test?', df_test.head())
