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

site_0 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site0.feather')
site_1 = np.load('/home/diogo/competitions/great-energy-predictor/data/data_leak_site1.pkl', allow_pickle=True)
site_2 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site2.feather')
site_4 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site4.feather')

site_0['site_id'] = 0
site_0['meter_reading'] = site_0['meter_reading_scraped']
site_0.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
site_0.dropna(inplace=True)
site_0.loc[site_0['meter_reading'] < 0, 'meter_reading'] = 0
site_0 = site_0.loc[(site_0['timestamp'].dt.year > 2016) &
                    (site_0['timestamp'].dt.year < 2019), :]

site_1['site_id'] = 1
site_1['meter_reading'] = site_1['meter_reading_scraped']
site_1.drop(['meter_reading_scraped'], axis=1, inplace=True)
site_1.dropna(inplace=True)
site_1.loc[site_1['meter_reading'] < 0, 'meter_reading'] = 0
site_1 = site_1.loc[(site_1['timestamp'].dt.year > 2016) &
                    (site_1['timestamp'].dt.year < 2019), :]

site_2['site_id'] = 2
site_2.dropna(inplace=True)
site_2.loc[site_2['meter_reading'] < 0, 'meter_reading'] = 0
site_2 = site_2.loc[(site_2['timestamp'].dt.year > 2016) &
                    (site_2['timestamp'].dt.year < 2019), :]

site_4['site_id'] = 4
site_4['meter_reading'] = site_4['meter_reading_scraped']
site_4['meter'] = 0
site_4.drop(['meter_reading_scraped'], axis=1, inplace=True)
site_4.dropna(inplace=True)
site_4.loc[site_4['meter_reading'] < 0, 'meter_reading'] = 0
site_4 = site_4.loc[(site_4['timestamp'].dt.year > 2016) &
                    (site_4['timestamp'].dt.year < 2019), :]

timezones = ['US/Eastern', 'Europe/London', 'US/Mountain', 'US/Pacific']
site_ids_leak = [0, 1, 2, 4]
for sid, tz in zip(site_ids_leak, timezones):
    leak = myutils.align_timestamp(leak, 0, strategy='timezone', site_id=sid, tz=tz)
del site_0, site_1, site_2, site_4
gc.collect()

subN = 11
sub = pd.read_feather('./submissions/submission_{:d}.feather'.format(subN))
test = pd.read_feather('./data/test_clean.feather')
test['meter_reading_pred'] = sub['meter_reading']
test = test[['building_id', 'meter', 'timestamp', 'meter_reading_pred']]
del sub

# Merge leak and test dataframes
leak = leak.merge(test, on=['building_id', 'meter', 'timestamp'], how='left')
leak = leak.dropna()
leak_score = np.sqrt(mean_squared_error(np.log1p(leak['meter_reading']), np.log1p(leak['meter_reading_pred'])))
print('Score on leaked data: {:.5f}'.format(leak_score))