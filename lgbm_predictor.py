#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:34:36 2019

@author: diogo
"""

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

def lgbm_model(X_train, y_train, X_val=None, y_val=None, nfolds=5):
    params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'rmse'},
                'subsample': 0.4,
                'subsample_freq': 1,
                'learning_rate': 0.25,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'lambda_l1': 1,
                'lambda_l2': 1
                }
    
    lgb_train = lgb.Dataset(X_train, y_train)
    if X_val is not None and y_val is not None:
        lgb_val = lgb.Dataset(X_val, y_val)
        sets = (lgb_train, lgb_val)
    else:
        sets=(lgb_train)
    model = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=sets,
                early_stopping_rounds=100,
                verbose_eval = 100)
    
    return model
