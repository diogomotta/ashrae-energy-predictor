#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 20:06:35 2019

@author: diogo
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import gc
import myutils
import itertools

# Control
importMySub1 = True
importMySub2 = True
importMySub3 = True
importMySub4 = True

combineSubs = False
combineBruteForce = False
combineGradDescAll = True
combineGradDescMeter = False

leak = pd.read_feather('./data/kaggle_leak.feather')
leak.fillna(0, inplace=True)
leak = leak[(leak['timestamp'].dt.year > 2016) & (leak['timestamp'].dt.year < 2019)]
leak.loc[leak['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
leak = leak[leak['building_id'] != 245]
leak = myutils.reduce_mem_usage(leak)

# Load test data
test = pd.read_feather('./data/test_clean.feather')
test = test[['building_id', 'meter', 'timestamp', 'row_id']]

# Load predictions data
if importMySub1:
    mysub_1 = pd.read_feather('./submissions/submission_30_noleak.feather') # 3-fold CV
if importMySub2:
    mysub_2 = pd.read_feather('./submissions/submission_31_noleak.feather') # 5-fold building_id grouping CV
if importMySub3:
    mysub_3 = pd.read_feather('./submissions/submission_32_noleak.feather') # 3-fold meter type grouping CV
if importMySub4:
    mysub_4 = pd.read_feather('./submissions/submission_33_noleak.feather') # 5-fold stratified month CV

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
del test
gc.collect()

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

# if combineBruteForce:
   
#     weight_range = list(np.linspace(0.2, 0.5, 51))
#     combs = [weight_range, weight_range]
#     # remember to do the reverse!
#     combs = list(itertools.product(*combs)) + list(itertools.product(*reversed(combs)))
#     combs = [comb for comb in combs if comb[0] + comb[1] > 0.92]
    
#     best_combi = list() # of the form (i, score)
#     for i, combi in enumerate(combs):
#         print('Iter {:d} out of {:d}'.format(i, len(combs)))
#         score1 = combi[0]
#         score2 = combi[1]
#         v = score1 * leak_mysub['mypred_1'].values + score2 * leak_mysub['mypred_2'].values
#         vl1p = np.log1p(v)
#         curr_score = np.sqrt(mean_squared_error(vl1p, np.log1p(leak_mysub['meter_reading'])))
        
#         if best_combi:
#             prev_score = best_combi[0][1]
#             if curr_score < prev_score:
#                 best_combi[:] = []
#                 best_combi += [(i, curr_score)]
#         else:
#             best_combi += [(i, curr_score)]
            
#     best_weights = combs[best_combi[0][0]]
#     print('best weights: {:} scoring {:.5f}'.format(best_weights, best_combi[0][1]))

if combineGradDescAll:   
    w0 = [0.25, 0.25, 0.25, 0.25]
    precision = 1e-4
    
    print('Optimizing subs combination weights')
    y_pred1 = leak_mysub['mypred_1'].values
    y_pred2 = leak_mysub['mypred_2'].values
    y_pred3 = leak_mysub['mypred_3'].values
    y_pred4 = leak_mysub['mypred_4'].values
    y_pred = [y_pred1, y_pred2, y_pred3, y_pred4]
    
    y_leak = leak_mysub['meter_reading'].values
    weights_opt, scores_opt = myutils.gradient_descent(w0, y_pred, y_leak, 0.1, 1000, precision)
        
    sub = pd.DataFrame()
    y_pred1 = leak_mysub['mypred_1'].values
    y_pred2 = leak_mysub['mypred_2'].values
    y_pred3 = leak_mysub['mypred_3'].values
    y_pred4 = leak_mysub['mypred_4'].values
    y_pred = [y_pred1, y_pred2, y_pred3, y_pred4]
    sub_comb = np.sum([w * y for w, y in zip(weights_opt, y_pred)], axis=0)
    sub = pd.DataFrame({'row_id': leak_mysub.index, 'meter_reading': sub_comb})
    
    y_leak = leak_mysub['meter_reading']
    sub = sub.sort_values(by=['row_id'])
    sub['meter_reading'] = np.clip(sub['meter_reading'].values, a_min=0, a_max=None)
    score_final = np.sqrt(mean_squared_error(np.log1p(sub['meter_reading'].values), np.log1p(y_leak)))
    print('Final score after weights optmization: {:.5f}'.format(score_final))

if combineGradDescMeter:   
    w0 = [0.25, 0.25, 0.25, 0.25]

    weights_opt = list()
    scores_opt = list()
    meter_opt = list()
    precision = [1e-4, 1e-3, 1e-6, 1e-4]
    for i, meter in enumerate(leak_mysub['meter'].unique()):
        print('Optimizing subs combination weights for meter {:d}'.format(int(meter)))
        y_pred1 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_1'].values
        y_pred2 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_2'].values
        y_pred3 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_3'].values
        y_pred4 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_4'].values
        y_pred = [y_pred1, y_pred2, y_pred3, y_pred4]
        
        y_leak = leak_mysub.loc[leak_mysub['meter']==meter, 'meter_reading']
        weights, score = myutils.gradient_descent(w0, y_pred, y_leak, 0.1, 1000, precision[i])
        weights_opt.append(weights)
        scores_opt.append(score)
        meter_opt.append(meter)
        
    sub = pd.DataFrame()
    for meter in leak_mysub['meter'].unique():
        row_id = leak_mysub.loc[leak_mysub['meter']==meter, :].index
        y_pred1 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_1'].values
        y_pred2 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_2'].values
        y_pred3 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_3'].values
        y_pred4 = leak_mysub.loc[leak_mysub['meter']==meter, 'mypred_4'].values
        y_pred = [y_pred1, y_pred2, y_pred3, y_pred4]
        weights = weights_opt[meter_opt==int(meter)]
        sub_comb = np.sum([w * y for w, y in zip(weights, y_pred)], axis=0)
        sub = pd.concat([sub, pd.DataFrame({'row_id': row_id, 'meter_reading': sub_comb})])
    
    y_leak = leak_mysub['meter_reading']
    sub = sub.sort_values(by=['row_id'])
    sub['meter_reading'] = np.clip(sub['meter_reading'].values, a_min=0, a_max=None)
    score_final = np.sqrt(mean_squared_error(np.log1p(sub['meter_reading'].values), np.log1p(y_leak)))
    print('Final score after weights optmization: {:.5f}'.format(score_final))

if combineSubs:
    # Load predictions data
    mysub_1 = pd.read_feather('./submissions/submission_30_leak.feather')
    mysub_2 = pd.read_feather('./submissions/submission_31_leak.feather')
    mysub_3 = pd.read_feather('./submissions/submission_32_leak.feather')
    mysub_4 = pd.read_feather('./submissions/submission_33_leak.feather')
    
    # Load test data
    test = pd.read_feather('./data/test_clean.feather')
    test = test[['building_id', 'meter', 'timestamp', 'row_id']]
    gc.collect()
    
    sub = pd.DataFrame()
    y_pred1 = mysub_1['meter_reading'].values
    y_pred2 = mysub_2['meter_reading'].values
    y_pred3 = mysub_3['meter_reading'].values
    y_pred4 = mysub_4['meter_reading'].values
    y_pred = [y_pred1, y_pred2, y_pred3, y_pred4]
    sub_comb = np.sum([w * y for w, y in zip(weights_opt, y_pred)], axis=0)
    sub = pd.DataFrame({'row_id': test.index, 'meter_reading': sub_comb})
    sub.to_csv('./submissions/submission_comb_30-31-32-33.csv', index=False)

