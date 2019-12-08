#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:42:08 2019

@author: diogo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import gc
import myutils
import itertools
import seaborn as sns

# Control
importKaggle = False
importMySub1 = True
importMySub2 = True
importMySub3 = False
importMySub4 = True
importMyLeak = False
importKaggleLeak = True

combineSubs = False
combineBruteForce = True
combineGradDesc = False

if importMyLeak:
    site_0 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site0.feather')
    site_1 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site1.feather')
    site_2 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site2.feather')
    site_4 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site4.feather')

    site_0['site_id'] = 0
    site_0['meter_reading'] = site_0['meter_reading_scraped']
    site_0.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
    site_0.fillna(0, inplace=True)
    site_0.loc[site_0['meter_reading'] < 0, 'meter_reading'] = 0
    site_0 = site_0.loc[(site_0['timestamp'].dt.year > 2016) &
                        (site_0['timestamp'].dt.year < 2019), :]
    
    site_1['site_id'] = 1
    site_1['meter_reading'] = site_1['meter_reading_scraped']
    site_1.drop(['meter_reading_scraped'], axis=1, inplace=True)
    site_1.fillna(0, inplace=True)
    site_1.loc[site_1['meter_reading'] < 0, 'meter_reading'] = 0
    site_1 = site_1.loc[(site_1['timestamp'].dt.year > 2016) &
                        (site_1['timestamp'].dt.year < 2019), :]
    
    site_2['site_id'] = 2
    site_2.fillna(0, inplace=True)
    site_2.loc[site_2['meter_reading'] < 0, 'meter_reading'] = 0
    site_2 = site_2.loc[(site_2['timestamp'].dt.year > 2016) &
                        (site_2['timestamp'].dt.year < 2019), :]
    
    site_4['site_id'] = 4
    site_4['meter_reading'] = site_4['meter_reading_scraped']
    site_4['meter'] = 0
    site_4.drop(['meter_reading_scraped'], axis=1, inplace=True)
    site_4.fillna(0, inplace=True)
    site_4.loc[site_4['meter_reading'] < 0, 'meter_reading'] = 0
    site_4 = site_4.loc[(site_4['timestamp'].dt.year > 2016) &
                        (site_4['timestamp'].dt.year < 2019), :]
    
    leak = pd.concat([site_0, site_1, site_2, site_4], sort=True)
    leak.reset_index(drop=True, inplace=True)
    del site_0, site_1, site_2, site_4
    gc.collect()
elif importKaggleLeak:
    leak = pd.read_feather('./data/kaggle_leak.feather')
    leak.fillna(0, inplace=True)
    leak = leak[(leak['timestamp'].dt.year > 2016) & (leak['timestamp'].dt.year < 2019)]
    leak.loc[leak['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
    leak = leak[leak['building_id'] != 245]
    leak = myutils.reduce_mem_usage(leak)

# Load predictions data
if importMySub1:
    mysub_1 = pd.read_feather('./submissions/submission_24_noleak.feather') # 2-fold by semester group CV
if importMySub2:
    mysub_2 = pd.read_feather('./submissions/submission_25_noleak.feather') # 3-fold CV
if importMySub3:
    mysub_3 = pd.read_feather('./submissions/submission_27_noleak.feather') # 3-fold by building_id CV
if importMySub4:
    mysub_4 = pd.read_feather('./submissions/submission_26_noleak.feather') # 2-fold CV splitting data by meter type
test = pd.read_feather('./data/test_clean.feather')
test = test[['building_id', 'meter', 'timestamp', 'row_id']]

if importKaggle:
#    kagglesub = pd.read_feather('./submissions/sub_halfandhalf.feather')
    kagglesub = pd.read_feather('./submissions/sub_3fold.feather')
    # create kaggle sub dataframe to merge
    kaggledf = test[['building_id', 'meter', 'timestamp', 'row_id']].copy()
    kaggledf['meter_reading_pred'] = kagglesub['meter_reading']
    del kagglesub

    # Merge leak and test dataframes
    leak_kaggle = leak.merge(kaggledf, on=['building_id', 'meter', 'timestamp'], how='left')
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak_kaggle['meter_reading']), np.log1p(leak_kaggle['meter_reading_pred'])))
    print('Kaggle score on leaked data: {:.5f}'.format(leak_score))
    sns.distplot(np.log1p(leak_kaggle['meter_reading_pred']))
    sns.distplot(np.log1p(leak_kaggle['meter_reading']))
    del leak_kaggle

# create my sub dataframe to merge
mydf = test[['building_id', 'meter', 'timestamp']].copy()
if importMySub1:
    mydf['mypred_1'] = mysub_1['meter_reading']
    del mysub_1
if importMySub2:
    mydf['mypred_2'] = mysub_2['meter_reading']
    del mysub_2
if importMySub3:
    mydf['mypred_3'] = mysub_3['meter_reading']
    del mysub_3
if importMySub4:
    mydf['mypred_4'] = mysub_4['meter_reading']
    del mysub_4

# Merge leak and test dataframes
leak_mysub = leak.merge(mydf, on=['building_id', 'meter', 'timestamp'], how='left')
leak_mysub = leak_mysub.dropna()

# Scoring models on leaked data
if importMySub1:
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak_mysub['meter_reading']), np.log1p(leak_mysub['mypred_1'])))
    print('Model 1 score on leaked data: {:.5f}'.format(leak_score))
    
if importMySub2:
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak_mysub['meter_reading']), np.log1p(leak_mysub['mypred_2'])))
    print('Model 2 score on leaked data: {:.5f}'.format(leak_score))

if importMySub3:
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak_mysub['meter_reading']), np.log1p(leak_mysub['mypred_3'])))
    print('Model 3 score on leaked data: {:.5f}'.format(leak_score)) 
    
if importMySub4:
    leak_score = np.sqrt(mean_squared_error(np.log1p(leak_mysub['meter_reading']), np.log1p(leak_mysub['mypred_4'])))
    print('Model 4 score on leaked data: {:.5f}'.format(leak_score))

if combineBruteForce:
   
    weight_range = list(np.linspace(0.2, 0.5, 51))
    combs = [weight_range, weight_range]
    # remember to do the reverse!
    combs = list(itertools.product(*combs)) + list(itertools.product(*reversed(combs)))
    combs = [comb for comb in combs if comb[0] + comb[1] > 0.92]
    
    best_combi = list() # of the form (i, score)
    for i, combi in enumerate(combs):
        print('Iter {:d} out of {:d}'.format(i, len(combs)))
        score1 = combi[0]
        score2 = combi[1]
        v = score1 * leak_mysub['mypred_1'].values + score2 * leak_mysub['mypred_2'].values
        vl1p = np.log1p(v)
        curr_score = np.sqrt(mean_squared_error(vl1p, np.log1p(leak_mysub['meter_reading'])))
        
        if best_combi:
            prev_score = best_combi[0][1]
            if curr_score < prev_score:
                best_combi[:] = []
                best_combi += [(i, curr_score)]
        else:
            best_combi += [(i, curr_score)]
            
    best_weights = combs[best_combi[0][0]]
    print('best weights: {:} scoring {:.5f}'.format(best_weights, best_combi[0][1]))

if combineSubs:
    # Load predictions data
    mysub_1 = pd.read_feather('./submissions/submission_24_leak.feather') # 3-fold CV
    mysub_2 = pd.read_feather('./submissions/submission_25_leak.feather') # 2-fold by semester group CV
    weights = best_weights
    sub_comb = weights[0] * mysub_1['meter_reading'].values + weights[1] * mysub_2['meter_reading'].values
    sub = pd.DataFrame({'row_id': test.index, 'meter_reading': sub_comb})
    sub.to_csv('./submissions/submission_comb_24-25.csv', index=False)
    # sub.to_feather('./submissions/submission_comb_24-25.feather')
    
if combineGradDesc:
    # Load predictions data
    mysub_1 = pd.read_feather('./submissions/submission_24_leak.feather') # 3-fold CV
    mysub_2 = pd.read_feather('./submissions/submission_25_leak.feather') # 2-fold by semester group CV
    
    w0 = [0.5, 0.5]
    
    myutils.gradient_descent(w0, [mysub_1, mysub_2], leak_mysub['meter_reading'], 0.1, 1000, 1e-1)



    
