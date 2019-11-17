#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:34:36 2019

@author: diogo
"""

import lightgbm as lgb
import numpy as np

def lgbm_model(X_train, y_train, X_val, y_val, params, cat_feats):
    
    model = lgb.LGBMRegressor(num_leaves=params['num_leaves'],
                            learning_rate=params['learning_rate'],
                            n_estimators=params['n_estimators'],
                            reg_alpha=params['lambda_l1'],
                            reg_lambda=params['lambda_l2'],
                            subsample=params['subsample'],
                            subsample_freq=params['subsample_freq'],
                            colsample_bytree=params['feature_fraction'])
    
    model.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='rmse',
            early_stopping_rounds=params['early_stopping_rounds'],
            categorical_feature=cat_feats,
            verbose=20)
    return model
