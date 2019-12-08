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
from sklearn.model_selection import train_test_split, GroupShuffleSplit, GroupKFold, KFold
import pickle
import gc

# My modules
import myutils
from lgbm_predictor import lgbm_model

# Models control
takeSample = False
doPred = False
splitOOF = False
plotImportance = True
groupBuildId = False
groupSemester = False
groupMeter = True
splitKFold = False

# Data leakage control
useMyLeak = False
useKaggleLeak = True
if useMyLeak:
    useSite0 = True
    useSite1 = True
    useSite2 = True
    useSite4 = True

subN = 26
train = pd.read_feather('./data/train_clean.feather')
if takeSample:
    train = train.sample(frac=0.1)
    train.reset_index(drop=True, inplace=True)
    gc.collect()
params = {
            'subsample': 0.8,
            'subsample_freq': 1,
            'learning_rate': 0.3,
            'num_leaves': 1023,
            'feature_fraction': 0.8,
            'lambda_l1': 0.5,
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
models_dict = {'models': list(), 'feat_importance': list(),
               'cv_results': list(), 'features': None, 'cv_type': None,
               'meter': list()}

if groupBuildId:
    print(40 * '-')
    NSPLITS = 3
    print('{:d}-fold group cross validation splitting data by building_id.'.format(NSPLITS))
    
    # Preparing data
    drop_cols = ['meter_reading', 'timestamp', 'date', 'square_feet', 'day', 'building_id',
                 'building_mean', 'building_std', 'building_median', 'site_id']
    categorical = ['primary_use', 'weekday', 'month', 'weekend', 'hour']
    feat_cols = [col for col in list(train) if col not in drop_cols]
    models_dict['features'] = feat_cols
    models_dict['cv_type'] = 'group_build_id'
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
        plt.savefig('./submissions/submission_{:d}.png'.format(subN))

if groupSemester:
    print(40 * '-')
    NSPLITS = 2
    print('2-fold group cross validation splitting data by semester.')

    # Preparing data
    train['semester'] = train['month'].replace({1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1,
                                                 7: 2, 8: 2, 9: 2, 10: 2, 11: 2, 12:2})

    drop_cols = ['meter_reading', 'timestamp', 'date', 'square_feet', 'day', 'semester', 'month',
                 'feels_like_temperature', 'relative_humidity', 'year_built', 'site_id', 'building_mean',
                 'building_std']
    categorical = ['primary_use', 'weekday', 'weekend', 'building_id', 'meter']
    feat_cols = [col for col in train.columns if col not in drop_cols]
    models_dict['features'] = feat_cols
    models_dict['cv_type'] = 'group_semester'
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
        ax.set_title('2-fold group validation splitting data by semester.')
        plt.savefig('./submissions/submission_{:d}.png'.format(subN))

if groupMeter:
    print(40 * '-')
    NSPLITS = 3
    print('3-fold group cross validation splitting data by meter type.')

    drop_cols = ['meter_reading', 'meter', 'timestamp', 'date', 'square_feet', 'day', 'month']
    categorical = ['primary_use', 'weekday', 'weekend', 'hour']
    feat_cols = [col for col in list(train) if col not in drop_cols]
    models_dict['features'] = feat_cols
    models_dict['cv_type'] = 'group_meter_semester'

    folds = KFold(n_splits=NSPLITS)
    for meter in train['meter'].unique():
        if meter == 0:
            params['learning_rate'] = 0.06
            params['lambda_l1'] = 0.5
        elif meter == 1:
            params['learning_rate'] = 0.03
            params['lambda_l1'] = 0.25
        elif meter == 2:
            params['learning_rate'] = 0.03
            params['lambda_l1'] = 0.1
        elif meter == 3:
            params['learning_rate'] = 0.03
            params['lambda_l1'] = 0.1
        
        print('Meter {:d}'.format(meter))
        train_meter = train.loc[train['meter']==meter, :]
        train_meter.reset_index(drop=True, inplace=True)
        X = train_meter[feat_cols]
        y = np.log1p(train_meter['meter_reading'])
        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
            model = lgbm_model(X.iloc[trn_idx, :], y[trn_idx],
                               X.iloc[val_idx, :], y[val_idx],
                               params,
                               cat_feats=categorical)
            models_dict['models'].append(model)
            models_dict['feat_importance'].append(model.feature_importances_)
            models_dict['cv_results'].append(model.best_score_)
            models_dict['meter'].append(meter)
            
    if plotImportance:
        plt_df = pd.DataFrame()
        for feat_imp in models_dict['feat_importance']:
            plt_df = plt_df.append(pd.DataFrame(zip(feat_imp, feat_cols), columns=['Value', 'Feature']))
        plt_df.reset_index(drop=True, inplace=True)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x='Value', y='Feature', data=plt_df.sort_values(by='Value', ascending=False))
        ax.set_title('2-fold group validation splitting data by meter type.')
        plt.savefig('./submissions/submission_{:d}.png'.format(subN))
        
if splitKFold:
    print(40 * '-')
    NSPLITS = 3
    print('3-fold cross validation.')

    drop_cols = ['meter_reading', 'timestamp', 'date', 'square_feet', 'day', 'month',
                 'building_mean', 'building_std', 'feels_like_temperature', 'year_built']
    categorical = ['primary_use', 'weekday', 'weekend', 'hour']
    feat_cols = [col for col in list(train) if col not in drop_cols]
    models_dict['features'] = feat_cols
    models_dict['cv_type'] = '3-fold'

    folds = KFold(n_splits=NSPLITS)
    X = train[feat_cols]
    y = np.log1p(train['meter_reading'])
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
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
        ax.set_title('3-fold validation.')
        plt.savefig('./submissions/submission_{:d}.png'.format(subN))

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
    test = pd.read_feather('./data/test_clean.feather')
    test = myutils.reduce_mem_usage(test)
    test['meter_reading'] = 0
    test['row_id_'] = 0
    feat_cols = models_dict['features']
    batch_size = 500000
    if groupMeter:
        for meter in np.unique(models_dict['meter']):
            print('Models for meter {:d}'.format(meter))
            models_idx = np.where(np.asarray(models_dict['meter']) == meter)[0]
            N_models = len(models_idx)
            
            for model_idx in models_idx:
                model = models_dict['models'][model_idx]
                predictions = list()
                test_meter = test.loc[test['meter']==meter, feat_cols]
                row_id_meter = test.loc[test['meter']==meter, 'row_id']
                test_meter.reset_index(drop=True, inplace=True)
                row_id_meter.reset_index(drop=True, inplace=True)
                N_BATCHES = int(len(test_meter)/batch_size) + 1
                for batch in range(N_BATCHES):
                    print('Predicting batch: {:d} of {:d}'.format(batch+1, N_BATCHES))
                    start_idx = batch*batch_size
                    end_idx = (batch+1)*batch_size - 1
                    print('Sample {:d} to {:d}'.format(start_idx, end_idx))
                    
                    X_batch = test_meter.loc[start_idx:end_idx, feat_cols]
                    predictions += list(np.expm1(model.predict(X_batch)))
                
                # Fill correspondend rows with predictions
                test.loc[row_id_meter.values, 'meter_reading'] += predictions
                
            # Divide by number of folds
            test['meter_reading'] = test['meter_reading'] / N_models
        
    else:
        for i, model in enumerate(models_dict['models']):
            feat_cols = models_dict['features']
            print('Model {:d}'.format(i))
            predictions = list()
            N_BATCHES = int(len(test)/batch_size)+1
            for batch in range(N_BATCHES):
                print('Predicting batch: {:d} of {:d}'.format(batch+1, N_BATCHES))
                start_idx = batch*batch_size
                end_idx = (batch+1)*batch_size - 1
                print('Sample {:d} to {:d}'.format(start_idx, end_idx))
                
                predictions += list(np.expm1(model.predict(test.loc[start_idx:end_idx, feat_cols])))
            test['meter_reading'] += predictions
        
        test['meter_reading'] = test['meter_reading'] / len(models_dict['models'])
        test['meter_reading'] = np.clip(test['meter_reading'].values, a_min=0, a_max=None)

    # Converting site_0 readings from meters 0 and 1 readings back to kBTU
    print('Converting meter 0 data in site 0 from kWh to kBTU...')
    rows_idx = test.loc[test['meter']==0, :].index
    test.loc[rows_idx, 'meter_reading'] = test.loc[rows_idx, 'meter_reading'] * 3.4118
    print('Done!')

    # Remove columns not used in merging
    test = test[['building_id', 'meter', 'timestamp', 'meter_reading']]
    
    sub = pd.DataFrame({'row_id': test.index, 'meter_reading': test['meter_reading']})
#    sub.to_csv('./submissions/submission_{:d}_noleak.csv'.format(subN), index=False)
    sub.to_feather('./submissions/submission_{:d}_noleak.feather'.format(subN))
    
    if useMyLeak:
        if useSite0:
            print('Inserting site_0 data...')
            site_0 = pd.read_feather('./data/data_leak_site0.feather')
            site_0['meter_reading'] = site_0['meter_reading_scraped']
            site_0.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
            site_0.dropna(inplace=True)
            site_0.loc[site_0['meter_reading'] < 0, 'meter_reading'] = 0
            site_0 = site_0.loc[(site_0['timestamp'].dt.year > 2016) &
                                (site_0['timestamp'].dt.year < 2019), :]
            test = myutils.replace_with_leaked(test, site_0)
            del site_0
            print('Done!')
              
        if useSite1:
            print('Inserting site_1 data...')
            site_1 = np.load('./data/data_leak_site1.pkl', allow_pickle=True)
            site_1['meter_reading'] = site_1['meter_reading_scraped']
            site_1.drop(['meter_reading_scraped'], axis=1, inplace=True)
            site_1.dropna(inplace=True)
            site_1.loc[site_1['meter_reading'] < 0, 'meter_reading'] = 0
            site_1 = site_1.loc[(site_1['timestamp'].dt.year > 2016) &
                                (site_1['timestamp'].dt.year < 2019), :]
            test = myutils.replace_with_leaked(test, site_1)
            del site_1
            print('Done!')
            
        if useSite2:
            print('Inserting site_2 data...')
            site_2 = pd.read_feather('./data/data_leak_site2.feather')
            site_2.dropna(inplace=True)
            site_2.loc[site_2['meter_reading'] < 0, 'meter_reading'] = 0
            site_2 = site_2.loc[(site_2['timestamp'].dt.year > 2016) &
                                (site_2['timestamp'].dt.year < 2019), :]
            test = myutils.replace_with_leaked(test, site_2)
            del site_2
            print('Done!')
            
        if useSite4:
            print('Inserting site_4 data...')
            site_4 = pd.read_feather('./data/data_leak_site4.feather')
            site_4['meter_reading'] = site_4['meter_reading_scraped']
            site_4['meter'] = 0
            site_4.drop(['meter_reading_scraped'], axis=1, inplace=True)
            site_4.dropna(inplace=True)
            site_4.loc[site_4['meter_reading'] < 0, 'meter_reading'] = 0
            site_4 = site_4.loc[(site_4['timestamp'].dt.year > 2016) &
                                (site_4['timestamp'].dt.year < 2019), :]
            test = myutils.replace_with_leaked(test, site_4)
            del site_4
            print('Done!')
    elif useKaggleLeak:
        leak = pd.read_feather('./data/kaggle_leak.feather')
        leak.fillna(0, inplace=True)
        leak = leak[(leak['timestamp'].dt.year > 2016) & (leak['timestamp'].dt.year < 2019)]
        leak.loc[leak['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
        leak = leak[leak['building_id'] != 245]
        leak = myutils.reduce_mem_usage(leak)
        test = myutils.replace_with_leaked(test, leak)
    
#    print('Converting meter 1 data in site 0 from kWh to kBTU...')
#    rows_idx = test.loc[test['meter']==1, :].index
#    test.loc[rows_idx, 'meter_reading'] = test.loc[rows_idx, 'meter_reading'] * 3.4118
#    print('Done!')
 
    gc.collect()
        
    sub = pd.DataFrame({'row_id': test.index, 'meter_reading': test['meter_reading']})
    sub.to_csv('./submissions/submission_{:d}_leak.csv'.format(subN), index=False)
    sub.to_feather('./submissions/submission_{:d}_leak.feather'.format(subN))
