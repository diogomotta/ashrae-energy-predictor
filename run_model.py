#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:43:12 2019

@author: diogo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
import seaborn as sns
import pickle

# My modules
import myutils
from lgbm_predictor import lgbm_model

train = pd.read_feather('./data/train_clean.feather')
test = pd.read_feather('./data/test_clean.feather')
sub = pd.read_csv('./data/sample_submission.csv')

print('Train dataframe')
train = myutils.reduce_mem_usage(train)
print('Test dataframe')
test = myutils.reduce_mem_usage(test)

#################################################################################
# Holdout set 1
# Split train set by building_id -> 20% to houldout
train_build, oof_build = train_test_split(train['building_id'].unique(), test_size=0.20)
oof_1 = train[train['building_id'].isin(oof_build)].reset_index(drop=True)
train = train[train['building_id'].isin(train_build)].reset_index(drop=True)

sns.countplot(x='site_id', data=train)
sns.countplot(x='site_id', data=oof_1)

drop_cols = ['meter_reading', 'timestamp']
feat_cols = [col for col in list(train) if col not in drop_cols]
X = train[feat_cols]
y = np.log1p(train['meter_reading'])

folds = KFold(n_splits=3, shuffle=True)
models = list()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
    model = lgbm_model(X.iloc[trn_idx, :], y[trn_idx],
                       X.iloc[val_idx, :], y[val_idx])
    models.append(model)

print(myutils.rmse(model.predict(oof_1[feat_cols]), np.log1p(oof_1['meter_reading'])))
pickle.dump(models, open('models.bin', 'wb'))

# doing the predictions
batch_size = 500000
predictions = []
for i, model in enumerate(models):
    print('Model {:d}'.format(i))
    predictions = list()
    for batch in range(int(len(test)/batch_size)+1):
        print('Predicting batch:', batch)
        predictions.append(np.expm1(model.predict(test[feat_cols].iloc[batch*batch_size:(batch+1)*batch_size])))
    sub['meter_reading'] += predictions
    
sub['meter_reading'] = sub['meter_reading'] / len(models)