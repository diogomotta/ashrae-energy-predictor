#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 23:43:12 2019

@author: diogo
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_squared_error
import pickle
import gc

# My modules
import myutils
from lgbm_predictor import lgbm_model

# Models control
doPred = False
splitOOF = False
plotImportance = True
groupBuildId = True
groupSemester = False

# Data leakage control
useSite0 = True
useSite1 = True
useSite2 = True
useSite4 = True

subN = 12
train = pd.read_feather('./data/train_clean.feather')
params = {
            'subsample': 0.8,
            'subsample_freq': 1,
            'learning_rate': 0.05,
            'num_leaves': 255,
            'feature_fraction': 0.8,
            'lambda_l1': 0.1,
            'lambda_l2': 0,
            'n_estimators': 2500,
            'early_stopping_rounds': 100
            }

#################################################################################
if splitOOF:
    # Holdout set 1
    # Split train set by building_id -> 10% to houldout
    train_build, oof_build = train_test_split(train['building_id'].unique(), test_size=0.10)
    oof_1 = train[train['building_id'].isin(oof_build)].reset_index(drop=True)
    train = train[train['building_id'].isin(train_build)].reset_index(drop=True)

#################################################################################
models_dict = {'models': list(), 'feat_importance': list(), 'cv_results': list()}

if groupBuildId:
    print(40 * '-')
    NSPLITS = 3
    print('{:d}-fold group cross validation splitting data by building_id.'.format(NSPLITS))
    
    # Preparing data
    drop_cols = ['meter_reading', 'timestamp', 'date', 'square_feet', 'day', 'building_id']
    categorical = ['primary_use', 'weekday', 'month', 'weekend', 'hour']
    feat_cols = [col for col in list(train) if col not in drop_cols]
    X = train[feat_cols]
    y = np.log1p(train['meter_reading'])
    
    group_cols = ['building_id']
    folds = GroupShuffleSplit(n_splits=NSPLITS)
    for col in group_cols:
        print('Splitting data by {:}'.format(col))
        split_group = train[col]
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_group)):
            model = lgbm_model(X.iloc[trn_idx, :], y[trn_idx],
                               X.iloc[val_idx, :], y[val_idx],
                               params,
                               cat_feats=categorical)
            models_dict['models'].append(model)
            models_dict['feat_importance'].append(model.feature_importances_)
            models_dict['cv_results'].append(model.best_score_)
          
    if plotImportance:
        plt_df = pd.DataFrame()
        for feat_imp in models_dict['feat_importance']:
            plt_df = plt_df.append(pd.DataFrame(zip(feat_imp, feat_cols), columns=['Value', 'Feature']))
        plt_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Value', y='Feature', data=plt_df.sort_values(by='Value', ascending=False))
        ax.set_title('{:d}-fold group validation splitting data by building_id.'.format(NSPLITS))

if groupSemester:
    print(40 * '-')
    NSPLITS = 2
    print('2-fold group cross validation splitting data by semester.')

    # Preparing data
    train['semester'] = train['month'].replace({1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
                                                 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12:2})

    drop_cols = ['meter_reading', 'timestamp', 'date', 'square_feet', 'day', 'semester', 'month']
    categorical = ['primary_use', 'weekday', 'weekend', 'hour']
    feat_cols = [col for col in list(train) if col not in drop_cols]
    X = train[feat_cols]
    y = np.log1p(train['meter_reading'])

    group_cols = ['semester']
    folds = GroupKFold(n_splits=NSPLITS)
    for col in group_cols:
        print('Splitting data by {:}'.format(col))
        split_group = train[col]
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y, groups=split_group)):
            print('Training Months:')
            print(np.unique(train.loc[trn_idx, 'month']))
            print('Validation Months:')
            print(np.unique(train.loc[val_idx, 'month']))
            model = lgbm_model(X.iloc[trn_idx, :], y[trn_idx],
                               X.iloc[val_idx, :], y[val_idx],
                               params,
                               cat_feats=categorical)
            models_dict['models'].append(model)
            models_dict['feat_importance'].append(model.feature_importances_)
            models_dict['cv_results'].append(model.best_score_)
            
    if plotImportance:
        plt_df = pd.DataFrame()
        for feat_imp in models_dict['feat_importance']:
            plt_df = plt_df.append(pd.DataFrame(zip(feat_imp, feat_cols), columns=['Value', 'Feature']))
        plt_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Value', y='Feature', data=plt_df.sort_values(by='Value', ascending=False))
        ax.set_title('2-fold group validation splitting data by month.')

if splitOOF:
    total_RMSE = list()
    for model in models_dict['models']:
        rmse = myutils.rmse(model.predict(oof_1[feat_cols]), np.log1p(oof_1['meter_reading']))
        total_RMSE.append(rmse)
        print('Model RMSE on OOF data: {:}'.format(rmse))
    print('Ensemble RMSE: {:.3f}'.format(np.asarray(total_RMSE).mean()))

pickle.dump(models_dict, open('./models/models_{:d}.bin'.format(subN), 'wb'))
del train, X, y
gc.collect()

# doing the predictions
if doPred:
    sub = pd.read_csv('./data/sample_submission.csv')
    test = pd.read_feather('./data/test_clean.feather')
    test = myutils.reduce_mem_usage(test)
    batch_size = 500000
    predictions = list()
    for i, model in enumerate(models_dict['models']):
        print('Model {:d}'.format(i))
        predictions = list()
        for batch in range(int(len(test)/batch_size)+1):
            print('Predicting batch:', batch)
            predictions += list(np.expm1(model.predict(test[feat_cols].iloc[batch*batch_size:(batch+1)*batch_size])))
        sub['meter_reading'] += predictions
        
    sub['meter_reading'] = sub['meter_reading'] / len(models_dict['models'])
    sub['meter_reading'] = np.clip(sub['meter_reading'].values, a_min=0, a_max=None)
    
    if useSite0:
        print('Inserting site_0 data...')
        site_0 = pd.read_feather('./data/data_leak_site0.feather')
        site_0['meter_reading'] = site_0['meter_reading_scraped']
        site_0.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
        site_0.fillna(0, inplace=True)
        site_0 = site_0.loc[(site_0['timestamp'].dt.year > 2016) &
                            (site_0['timestamp'].dt.year < 2019), :]
        site_0.loc[site_0['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
        site_0.dropna(inplace=True)
        
        leak_score = 0
        for bid in site_0['building_id'].unique():
            temp_df = site_0.loc[(site_0['building_id']==bid), ['meter', 'meter_reading']]
            for m in temp_df['meter'].unique():
                v0 = sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'].values
                v1 = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
                leak_score += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)
                
                sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'] = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
        del site_0
        print('Done!')
          
    if useSite1:
        print('Inserting site_1 data...')
        site_1 = np.load('./data/data_leak_site1.pkl', allow_pickle=True)
        site_1['meter_reading'] = site_1['meter_reading_scraped']
        site_1.drop(['meter_reading_scraped'], axis=1, inplace=True)
        site_1.fillna(0, inplace=True)
        site_1 = site_1.loc[(site_1['timestamp'].dt.year > 2016) &
                            (site_1['timestamp'].dt.year < 2019), :]
        site_1.loc[site_1['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
        site_1.dropna(inplace=True)
        
        leak_score = 0
        for bid in site_1['building_id'].unique():
            temp_df = site_1.loc[(site_1['building_id']==bid), ['meter', 'meter_reading']]
            for m in temp_df['meter'].unique():
                v0 = sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'].values
                v1 = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
                leak_score += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)
                
                sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'] = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
        del site_1
        print('Done!')
        
    if useSite2:
        print('Inserting site_2 data...')
        site_2 = pd.read_feather('./data/data_leak_site2.feather')
        site_2.fillna(0, inplace=True)
        site_2 = site_2.loc[(site_2['timestamp'].dt.year > 2016) &
                            (site_2['timestamp'].dt.year < 2019), :]
        site_2.loc[site_2['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
        site_2.dropna(inplace=True)
        
        leak_score = 0
        for bid in site_2['building_id'].unique():
            temp_df = site_2.loc[(site_2['building_id']==bid), ['meter', 'meter_reading']]
            for m in temp_df['meter'].unique():
                v0 = sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'].values
                v1 = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
                leak_score += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)
                
                sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'] = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
        del site_2
        print('Done!')
        
    if useSite4:
        print('Inserting site_4 data...')
        site_4 = pd.read_feather('./data/data_leak_site4.feather')
        site_4.fillna(0, inplace=True)
        site_4 = site_4.loc[(site_4['timestamp'].dt.year > 2016) &
                            (site_4['timestamp'].dt.year < 2019), :]
        site_4.loc[site_4['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
        site_4.dropna(inplace=True)
        
        leak_score = 0
        for bid in site_4['building_id'].unique():
            temp_df = site_4.loc[(site_4['building_id']==bid), ['meter', 'meter_reading']]
            for m in temp_df['meter'].unique():
                v0 = sub.loc[(test['building_id']==bid) & (test['meter']==m), 'meter_reading'].values
                v1 = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
#                leak_score += mean_squared_error(np.log1p(v0), np.log1p(v1)) * len(v0)
                
                sub.loc[(test['building_id']==bid) &
                        (test['meter']==m), 'meter_reading'] = temp_df.loc[temp_df['meter']==m, 'meter_reading'].values
        del site_4
        print('Done!')
 
        gc.collect()
        
    sub.to_csv('submission_{:d}.csv'.format(subN), index=False)
