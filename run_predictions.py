#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:44:54 2019

@author: diogo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc

# My modules
import myutils

# Data leakage control
useMyLeak = False
useKaggleLeak = False
if useMyLeak:
    useSite0 = True
    useSite1 = True
    useSite2 = True
    useSite4 = True

# save prediction
savePred = True

# Load Model
subN = 21
models_dict = np.load('./models/models_{:d}.bin'.format(subN), allow_pickle=True)
feat_cols = models_dict['features']

test = pd.read_feather('./data/test_clean.feather')
test = myutils.reduce_mem_usage(test)
test['meter_reading'] = 0
batch_size = 500000
predictions = list()
for i, model in enumerate(models_dict['models']):
    print('Model {:d}'.format(i))
    predictions = list()
    for batch in range(int(len(test)/batch_size)+1):
        print('Predicting batch:', batch)
        predictions += list(np.expm1(model.predict(test[feat_cols].iloc[batch*batch_size:(batch+1)*batch_size])))
    test['meter_reading'] += predictions
    
test['meter_reading'] = test['meter_reading'] / len(models_dict['models'])
test['meter_reading'] = np.clip(test['meter_reading'].values, a_min=0, a_max=None)

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

gc.collect()

if savePred:
    if useMyLeak or useKaggleLeak:
        sub = pd.DataFrame({'row_id': test.index, 'meter_reading': test['meter_reading']})
    #    sub.to_csv('./submissions/submission_{:d}_noleak.csv'.format(subN), index=False)
        sub.to_feather('./submissions/submission_{:d}_leak.feather'.format(subN))
    else:
        sub = pd.DataFrame({'row_id': test.index, 'meter_reading': test['meter_reading']})
    #    sub.to_csv('./submissions/submission_{:d}_noleak.csv'.format(subN), index=False)
        sub.to_feather('./submissions/submission_{:d}_noleak.feather'.format(subN))