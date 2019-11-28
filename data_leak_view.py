#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 11:25:49 2019

@author: diogo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import myutils

train = pd.read_feather('./data/train.feather')
site0 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site0.feather')
site1 = np.load('/home/diogo/competitions/great-energy-predictor/data/data_leak_site1.pkl', allow_pickle=True)
site2 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site2.feather')
site4 = pd.read_feather('/home/diogo/competitions/great-energy-predictor/data/data_leak_site4.feather')

# plot control
plot_site0 = False
plot_site1 = False
plot_site2 = False
plot_site4 = True

if plot_site0:
    print('Loading site_0 leaked data...')
    site0['meter_reading'] = site0['meter_reading_scraped']
    site0.drop(['meter_reading_scraped'], axis=1, inplace=True)
    site0.dropna(inplace=True)
    site0.loc[site0['meter_reading'] < 0, 'meter_reading'] = 0
    site0 = site0[(site0['timestamp'].dt.year == 2016)]
    
    print('Replacing original data with leaked data...')
    train_rplc = myutils.replace_with_leaked(train, site0)
    print('Done!')
    
    # site0 plot
    month_plt = 9
    year_plt = 2016
    for bid in site0['building_id'].sample(3):
        plt.figure()
        train_plt = train.loc[(train['building_id'] == bid) & 
                              (train['timestamp'].dt.month == month_plt) & 
                              (train['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        leak_plt = site0.loc[(site0['building_id'] == bid) & 
                               (site0['timestamp'].dt.month == month_plt) &
                               (site0['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        train_rplc_plt = train_rplc.loc[(train_rplc['building_id'] == bid) & 
                                       (train_rplc['timestamp'].dt.month == month_plt) &
                                       (train_rplc['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        plt.plot(train_plt['timestamp'], train_plt['meter_reading'], label='train')    
        plt.plot(leak_plt['timestamp'], leak_plt['meter_reading'], '.',label='leak')
        plt.plot(train_rplc_plt['timestamp'], train_rplc_plt['meter_reading'], 'o',label='train with replaced leak data')
        plt.legend()

if plot_site1:
    print('Loading site_1 leaked data...')
    site1['meter_reading'] = site1['meter_reading_scraped']
    site1.drop(['meter_reading_scraped'], axis=1, inplace=True)
    site1.dropna(inplace=True)
    site1.loc[site1['meter_reading'] < 0, 'meter_reading'] = 0
    site1 = site1[(site1['timestamp'].dt.year == 2016)]
    
    print('Replacing original data with leaked data...')
    train_rplc = myutils.replace_with_leaked(train, site1)
    print('Done!')
    
    # site1 plot
    month_plt = 9
    year_plt = 2016
    for bid in site1['building_id'].sample(3):
        plt.figure()
        train_plt = train.loc[(train['building_id'] == bid) & 
                              (train['timestamp'].dt.month == month_plt) & 
                              (train['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        leak_plt = site1.loc[(site1['building_id'] == bid) & 
                               (site1['timestamp'].dt.month == month_plt) &
                               (site1['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        train_rplc_plt = train_rplc.loc[(train_rplc['building_id'] == bid) & 
                                       (train_rplc['timestamp'].dt.month == month_plt) &
                                       (train_rplc['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        plt.plot(train_plt['timestamp'], train_plt['meter_reading'], label='train')    
        plt.plot(leak_plt['timestamp'], leak_plt['meter_reading'], '.', label='leak')
        plt.plot(train_rplc_plt['timestamp'], train_rplc_plt['meter_reading'], 'o', label='train with replaced leak data')
        plt.legend()

if plot_site2:
    print('Loading site_2 leaked data...')
    site2.dropna(inplace=True)
    site2.loc[site2['meter_reading'] < 0, 'meter_reading'] = 0
    site2 = site2[(site2['timestamp'].dt.year == 2016)]
    
    print('Replacing original data with leaked data...')
    train_rplc = myutils.replace_with_leaked(train, site2)
    print('Done!')
    
    # site2 plot
    month_plt = 9
    year_plt = 2016
    for bid in site2['building_id'].sample(3):
        plt.figure()
        train_plt = train.loc[(train['building_id'] == bid) & 
                              (train['timestamp'].dt.month == month_plt) & 
                              (train['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        leak_plt = site2.loc[(site2['building_id'] == bid) & 
                               (site2['timestamp'].dt.month == month_plt) &
                               (site2['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        train_rplc_plt = train_rplc.loc[(train_rplc['building_id'] == bid) & 
                                       (train_rplc['timestamp'].dt.month == month_plt) &
                                       (train_rplc['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        plt.plot(train_plt['timestamp'], train_plt['meter_reading'], label='train')    
        plt.plot(leak_plt['timestamp'], leak_plt['meter_reading'], '.', label='leak')
        plt.plot(train_rplc_plt['timestamp'], train_rplc_plt['meter_reading'], 'o', label='train with replaced leak data')
        plt.legend()

if plot_site4:
    print('Loading site_4 leaked data...')
    site4['meter_reading'] = site4['meter_reading_scraped']
    site4['meter'] = 0
    site4.drop(['meter_reading_scraped'], axis=1, inplace=True)
    site4.dropna(inplace=True)
    site4.loc[site4['meter_reading'] < 0, 'meter_reading'] = 0
    site4 = site4[(site4['timestamp'].dt.year == 2016)]
    
    print('Replacing original data with leaked data...')
    train_rplc = myutils.replace_with_leaked(train, site4)
    print('Done!')
    
    # site4 plot
    month_plt = 9
    year_plt = 2016
    for bid in site4['building_id'].sample(3):
        plt.figure()
        train_plt = train.loc[(train['building_id'] == bid) & 
                              (train['timestamp'].dt.month == month_plt) & 
                              (train['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        leak_plt = site4.loc[(site4['building_id'] == bid) & 
                               (site4['timestamp'].dt.month == month_plt) &
                               (site4['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        train_rplc_plt = train_rplc.loc[(train_rplc['building_id'] == bid) & 
                                       (train_rplc['timestamp'].dt.month == month_plt) &
                                       (train_rplc['timestamp'].dt.year == year_plt), ['timestamp', 'meter_reading']]
        plt.plot(train_plt['timestamp'], train_plt['meter_reading'], label='train')    
        plt.plot(leak_plt['timestamp'], leak_plt['meter_reading'], '.', label='leak')
        plt.plot(train_rplc_plt['timestamp'], train_rplc_plt['meter_reading'], 'o', label='train with replaced leak data')
        plt.legend()