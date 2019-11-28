# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import gc
from sklearn.preprocessing import LabelEncoder

# My modules
import myutils

dataset = 'test'

calcOffset = False
alignTimestamps = True
encodeCyclic = False
interpMissing = False
removeOutliers = True
addLag = False

# Leaked data
useSite0 = True
useSite1 = False
useSite2 = False
useSite4 = False

print('Loading data...')
if dataset == 'train':
    meter_train = pd.read_feather('./data/train.feather')
elif dataset == 'test':
    meter_data = pd.read_feather('./data/test.feather')
    meter_train = pd.read_feather('./data/train.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')
weather_test = pd.read_feather('./data/weather_test.feather')
print('Done!')

# Removing bad data in Site 0
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588
print('Removing bad data from site 0...')
meter_train = meter_train.loc[(meter_train['building_id'] > 104) | (meter_train['timestamp'] > '2016-05-20'), :]
print('Done!')

# Removing bad data in building 1099 on meter 2 (steam)
print('Removing bad data from building 1099...')
rows_idx = meter_train.loc[(meter_train['building_id'] == 1099) & (meter_train['meter'] == 2), :].index
meter_train.loc[rows_idx, 'meter_reading'] = meter_train.loc[rows_idx, 'meter_reading'] / 1e3
print('Done!')

if removeOutliers:
    # Removing outliers in reading data
    print('Clipping outliers above 99% quantile')
    meter_train_group = meter_train.groupby(['building_id', 'meter'])['meter_reading']
    meter_train['meter_reading'] = meter_train_group.apply(lambda x: x.clip(lower=0, upper=x.quantile(0.99)))
    print('Done!')
    del meter_train_group

if dataset == 'train':
    meter_data = meter_train
    del meter_train
    
print('Calculating median reading for each building...')
if dataset == 'test':
    meter_train_group = meter_train.groupby(['building_id', 'meter'])['meter_reading']
    del meter_train
else:
    meter_train_group = meter_data.groupby(['building_id', 'meter'])['meter_reading']

building_median = np.log1p(meter_train_group.median()).astype(np.float32)
del meter_train_group
print('Done!')

print('Encoding building metadata and merging to meter data...')
# Encoding building metadata
building['primary_use'] = building['primary_use'].astype('category')
label = LabelEncoder()
building['primary_use'] = label.fit_transform(building['primary_use']).astype('uint8')
# Replacing NaNs
building['floor_count'] = building['floor_count'].fillna(0).astype('uint8')
# Including ln of square feet area of the building
building['log_square_feet'] = np.log(building['square_feet']).astype('float32')

# Merging building metadata and train/test sets
meter_data = meter_data.merge(building, on='building_id', how='left')
del building
print('Done!')

if calcOffset:
    print('Calculating timestamp correcting offset...')
    # Find timestamps alignment offset
    weather = pd.concat([weather_train, weather_test],ignore_index=True)
    offset = myutils.calculate_time_offset(weather)
    if dataset == 'train':
        del weather, weather_test
    elif dataset == 'test':
        del weather, weather_train
    gc.collect()
    print('Done!')
else:
    site_GMT_offsets = [5, 0, 7, 5, 8, 0, 5, 5, 5, 6, 7, 5, 0, 6, 5, 5]
    offset = pd.Series(site_GMT_offsets)
    offset.index.name = 'site_id'
    
if dataset == 'train':
    weather = weather_train
    del weather_train
elif dataset == 'test':
    weather = weather_test
    del weather_test

if alignTimestamps:
    print('Aligning weather timestamps..')
    # Align timestamp offset
    ## Using actual timezones from leaked data
    #timezones = ['US/Eastern', 'Europe/London', 'US/Mountain', 'US/Pacific']
    #site_ids_leak = [0, 1, 2, 4]
    #for sid, tz in zip(site_ids_leak, timezones):
    #    all_data = myutils.align_timestamp(all_data, 0, strategy='timezone', site_id=sid, tz=tz)
    
    # Using 14h calculation
    weather = myutils.align_timestamp(weather, offset, strategy='14h_calc')
    print('Done!')

print('Adding new features to weather dataframe')
# Find missing dates
weather = myutils.find_missing_dates(weather)

# Replace NaNs with interpolation
nan_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
            'sea_level_pressure', 'wind_direction', 'wind_speed']
weather = myutils.replace_nan(weather, nan_cols, 'interp')

# Replace remaining NaNs with daily mean value
nan_cols = weather.loc[:, weather.isnull().any()].columns

# Add date column
weather['date'] = weather['timestamp'].dt.date
weather = myutils.replace_nan(weather, nan_cols, 'mean_group', group=['site_id', 'date'])

# Replace remaining NaNs with daily mean value
nan_cols = weather.loc[:, weather.isnull().any()].columns

weather = myutils.replace_nan(weather, nan_cols, 'median')

# Add relative humidity and feels like temperature
weather = myutils.calculate_relative_humidity(weather)
weather = myutils.calculate_feels_like_tempeature(weather)
if addLag:
    lag_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                'sea_level_pressure', 'wind_direction', 'wind_speed']
    weather = myutils.create_lag_features(weather, lag_cols, 'site_id', 72, 'mean')
print('Done!')

print('Saving data to feather...')
if dataset == 'train':
    weather.to_feather('./data/weather_train_feats.feather')
elif dataset == 'test':
    weather.to_feather('./data/weather_test_feats.feather')
print('Done!')
  
print('Merging weather and meter dataframes...')
# Merge weather data
all_data = meter_data.merge(weather, on=['site_id', 'timestamp'], how='left')
all_data = myutils.reduce_mem_usage(all_data)
del weather, meter_data
gc.collect()
print('Done!')

print('Adding new date time features...')
all_data = myutils.preprocess_datetime(all_data, date_feat=['h', 'd', 'w', 'm', 'wk'])
if encodeCyclic:
    all_data = myutils.encode_cyclic_feature(all_data, 'weekday', 7)
    all_data = myutils.encode_cyclic_feature(all_data, 'hour', 24)
    all_data = myutils.encode_cyclic_feature(all_data, 'day', 31)
    all_data = myutils.encode_cyclic_feature(all_data, 'month', 12)
all_data = myutils.reduce_mem_usage(all_data)
all_data['age'] = all_data['timestamp'].dt.year - all_data['year_built']
all_data['age'] = all_data['age'].fillna(-99).astype(np.int16)
all_data['year_built'] = all_data['year_built'].fillna(-99).astype(np.int16)
print('Done!')

print('Adding median building reading per meter...')
all_data['building_median'] = all_data[['building_id', 'meter']].merge(building_median, on=['building_id', 'meter'], how='left')['meter_reading']
print('Done!')

if interpMissing:
    print('Interpolating missing data...')
    # interpolate missing values using a cubic spline
    nan_cols = all_data.loc[:, all_data.isnull().any()].columns
    all_data = myutils.replace_nan(all_data, nan_cols, 'interp')
    print('Done!')

if dataset == 'train':
    if useSite0:
        print('Loading site_0 leaked data...')
        site_0 = pd.read_feather('./data/data_leak_site0.feather')
        site_0['meter_reading'] = site_0['meter_reading_scraped']
        site_0.drop(['meter_reading_original','meter_reading_scraped'], axis=1, inplace=True)
        site_0.dropna(inplace=True)
        site_0.loc[site_0['meter_reading'] < 0, 'meter_reading'] = 0
        site_0 = site_0[(site_0['timestamp'].dt.year == 2016)]
        
        print('Replacing original data with leaked data...')
        all_data = myutils.replace_with_leaked(all_data, site_0)
        print('Done!')
        del site_0
    
    if useSite1:
        print('Loading site_1 leaked data...')
        site_1 = np.load('./data/data_leak_site1.pkl', allow_pickle=True)
        site_1['meter_reading'] = site_1['meter_reading_scraped']
        site_1.drop(['meter_reading_scraped'], axis=1, inplace=True)
        site_1.dropna(inplace=True)
        site_1.loc[site_1['meter_reading'] < 0, 'meter_reading'] = 0
        site_1 = site_1[(site_1['timestamp'].dt.year == 2016)]
        
        print('Replacing original data with leaked data...')
        all_data = myutils.replace_with_leaked(all_data, site_1)
        print('Done!')
        del site_1
        
    if useSite2:
        print('Loading site_2 leaked data...')
        site_2 = pd.read_feather('./data/data_leak_site2.feather')
        site_2.dropna(inplace=True)
        site_2.loc[site_2['meter_reading'] < 0, 'meter_reading'] = 0
        site_2 = site_2[(site_2['timestamp'].dt.year == 2016)]
        
        print('Replacing original data with leaked data...')
        all_data = myutils.replace_with_leaked(all_data, site_2)
        print('Done!')
        del site_2
        
    if useSite4:
        print('Loading site_4 leaked data...')
        site_4 = pd.read_feather('./data/data_leak_site4.feather')
        site_4['meter_reading'] = site_4['meter_reading_scraped']
        site_4['meter'] = 0
        site_4.drop(['meter_reading_scraped'], axis=1, inplace=True)
        site_4.dropna(inplace=True)
        site_4.loc[site_4['meter_reading'] < 0, 'meter_reading'] = 0
        site_4 = site_4[(site_4['timestamp'].dt.year == 2016)]
        
        print('Replacing original data with leaked data...')
        all_data = myutils.replace_with_leaked(all_data, site_4)
        print('Done!')
        del site_4

print('Saving data to feather...')
if dataset == 'train':
    all_data = myutils.convert_float16_to_float32(all_data)
    all_data.to_feather('./data/train_clean.feather')
elif dataset == 'test':
    all_data = myutils.convert_float16_to_float32(all_data)
    all_data.to_feather('./data/test_clean.feather')
print('Done!')

#del all_data
#gc.collect()
