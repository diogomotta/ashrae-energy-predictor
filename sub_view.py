#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 18:15:38 2019

@author: diogo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc

sub_27noleak = pd.read_feather('./submissions/submission_27_noleak.feather')
sub_noleak = pd.read_feather('./submissions/submission_30_noleak.feather')
# sub_leak = pd.read_feather('./submissions/submission_30_leak.feather')
sub_leak = pd.read_csv('./submissions/submission_comb_30-31-32-33.csv')
test = pd.read_feather('./data/test_clean.feather')

#score = mean_squared_error(np.log1p(sub_noleak['meter_reading']), np.log1p(sub_leak['meter_reading']))
#print('sub 18 with leak vs sub 18 with no leak: {:.5f}'.format(score))
#
score = mean_squared_error(np.log1p(sub_noleak['meter_reading']), np.log1p(sub_27noleak['meter_reading']))
print('sub 27 vs sub 30 with no leak: {:.5f}'.format(score))


site_id = 7
meter = 0
plt.close('all')
for bid in np.random.choice(test.loc[test['site_id']==site_id, 'building_id'].unique(), 5):
    plt_idx = test.loc[(test['building_id']==bid) & 
                       (test['meter']==0), :].index

    plt.figure()
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_noleak.loc[plt_idx, 'meter_reading'], '.', label='without leaked data')    
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_leak.loc[plt_idx, 'meter_reading'], '-', label='with leaked data')  
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_27noleak.loc[plt_idx, 'meter_reading'], '-', label='submission 27')    
    plt.legend()
    plt.title('Site {:d} - Building {:d}'.format(site_id, bid))