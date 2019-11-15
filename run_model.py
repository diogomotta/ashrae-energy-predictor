#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:43:12 2019

@author: diogo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupKFold
import pickle
import gc

# My modules
import myutils
from lgbm_predictor import lgbm_model

doPred = True
subN = 4
train = pd.read_feather('./data/train_clean.feather')

params = {
            'subsample': 0.4,
            'subsample_freq': 1,
            'learning_rate': 0.2,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'lambda_l1': 0.3,
            'lambda_l2': 0,
            'n_estimators': 1000,
            'early_stopping_rounds': 20
            }

#################################################################################
# Holdout set 1
# Split train set by building_id -> 10% to houldout
train_build, oof_build = train_test_split(train['building_id'].unique(), test_size=0.10)
oof_1 = train[train['building_id'].isin(oof_build)].reset_index(drop=True)
train = train[train['building_id'].isin(train_build)].reset_index(drop=True)

# Preparing data
drop_cols = ['meter_reading', 'timestamp', 'date', 'building_id', 'site_id', 'square_feet']
categorical = ['primary_use', 'day', 'weekday', 'month', 'weekend']
feat_cols = [col for col in list(train) if col not in drop_cols]
X = train[feat_cols]
y = np.log1p(train['meter_reading'])

#################################################################################
group_cols = ['month']
folds = GroupKFold(n_splits=3)
models = list()

for col in group_cols:
    print('Splitting data by {:}'.format(col))
    split_group = train[col]
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_group)):
        model = lgbm_model(X.iloc[trn_idx, :], y[trn_idx],
                           X.iloc[val_idx, :], y[val_idx],
                           params,
                           cat_feats=categorical)
        models.append(model)

total_RMSE = list()
for model in models:
    rmse = myutils.rmse(model.predict(oof_1[feat_cols]), np.log1p(oof_1['meter_reading']))
    total_RMSE.append(rmse)
    print('Model RMSE on OOF data: {:}'.format(rmse))

print('Ensemble RMSE: {:.3f}'.format(np.asarray(total_RMSE).mean()))
pickle.dump(models, open('./models/models_{:d}.bin'.format(subN), 'wb'))
del train, X, y
gc.collect()

# doing the predictions
if doPred:
    sub = pd.read_csv('./data/sample_submission.csv')
    test = pd.read_feather('./data/test_clean.feather')
    test = myutils.reduce_mem_usage(test)
    test = test[feat_cols]
    batch_size = 250000
    predictions = list()
    for i, model in enumerate(models):
        print('Model {:d}'.format(i))
        predictions = list()
        for batch in range(int(len(test)/batch_size)+1):
            print('Predicting batch:', batch)
            predictions += list(np.expm1(model.predict(test[feat_cols].iloc[batch*batch_size:(batch+1)*batch_size])))
        sub['meter_reading'] += predictions
        
    sub['meter_reading'] = sub['meter_reading'] / len(models)
    sub['meter_reading'] = np.clip(sub['meter_reading'].values, a_min=0, a_max=None)
    sub.to_csv('submission_{:d}.csv'.format(subN), index=False)