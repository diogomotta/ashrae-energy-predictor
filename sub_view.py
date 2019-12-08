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

sub_noleak = pd.read_feather('./submissions/submission_18_noleak.feather')
sub_leak = pd.read_feather('./submissions/submission_leak_18.feather')
sub_11 = pd.read_feather('./submissions/submission_11.feather')
sub_15 = pd.read_feather('./submissions/submission_15.feather')
test = pd.read_feather('./data/test_clean.feather')

#score = mean_squared_error(np.log1p(sub_noleak['meter_reading']), np.log1p(sub_leak['meter_reading']))
#print('sub 18 with leak vs sub 18 with no leak: {:.5f}'.format(score))
#
score = mean_squared_error(np.log1p(sub_noleak['meter_reading']), np.log1p(sub_11['meter_reading']))
print('sub 11 vs sub 18 with no leak: {:.5f}'.format(score))

#score = mean_squared_error(np.log1p(sub_leak['meter_reading']), np.log1p(sub_11['meter_reading']))
#print('sub 11 vs sub 18 with leak: {:.5f}'.format(score))

score = mean_squared_error(np.log1p(sub_11['meter_reading']), np.log1p(sub_15['meter_reading']))
print('sub 11 vs sub 15 leak: {:.5f}'.format(score))


site_id = 12
plt.close('all')
for bid in np.random.choice(test.loc[test['site_id']==site_id, 'building_id'].unique(), 5):
    plt_idx = test.loc[test['building_id']==bid, :].index

    plt.figure()
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_noleak.loc[plt_idx, 'meter_reading'], '.', label='without leaked data')    
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_leak.loc[plt_idx, 'meter_reading'], '-', label='with leaked data')    
    plt.plot(test.loc[plt_idx, 'timestamp'], sub_11.loc[plt_idx, 'meter_reading'], '--', label='submission 11')
    plt.legend()
    plt.title('Site {:d} - Building {:d}'.format(site_id, bid))