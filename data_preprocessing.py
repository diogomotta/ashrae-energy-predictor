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

encodeCyclic = False
interpMissing = False

print('Loading data...')
if dataset == 'train':
    meter_data = pd.read_feather('./data/train.feather')
elif dataset == 'test':
    meter_data = pd.read_feather('./data/test.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')
weather_test = pd.read_feather('./data/weather_test.feather')
print('Done!')

if dataset == 'train':
    # Removing bad data in Site 0
    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588
    print('Removing bad data from site 0...')
    meter_data = meter_data.loc[(meter_data['building_id'] > 104) | (meter_data['timestamp'] > '2016-05-20'), :]
    print('Done!')
    
    # Removing bad data in building 1099 on meter 2 (steam)
    print('Removing bad data from building 1099...')
    rows_idx = meter_data.loc[(meter_data['building_id'] == 1099) & (meter_data['meter'] == 2), :].index
    meter_data.loc[rows_idx, 'meter_reading'] = meter_data.loc[rows_idx, 'meter_reading'] / 1e3

print('Encoding building metadata and merging to meter data...')
# Encoding building metadata
building['primary_use'] = building['primary_use'].astype('category')
label = LabelEncoder()
building['primary_use'] = label.fit_transform(building['primary_use']).astype('uint8')
# Replacing NaNs
building['floor_count'] = building['floor_count'].fillna(0).astype('uint8')
building['year_built'] = building['year_built'].fillna(-999).astype('uint8')
# Including ln of square feet area of the building
building['log_square_feet'] = np.log(building['square_feet']).astype('float32')

# Merging building metadata and train/test sets
meter_data = meter_data.merge(building, on='building_id', how='left')
del building
print('Done!')

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

print('Adding new features to weather dataframe')
# Add some new weather features
if dataset == 'train':
    weather = weather_train
    del weather_train
elif dataset == 'test':
    weather = weather_test
    del weather_test

# Replace NaNs with interpolation
nan_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
            'sea_level_pressure', 'wind_direction', 'wind_speed']
weather = myutils.replace_nan(weather, nan_cols, 'interp')

# Replace remaining NaNs with daily mean value
nan_cols = weather.loc[:, weather.isnull().any()].columns

# Add date column
weather['date'] = weather['timestamp'].dt.date
weather = myutils.replace_nan(weather, nan_cols, 'mean_group', group=['site_id', 'date'])

# Add relative humidity and feels like temperature
weather = myutils.calculate_relative_humidity(weather)
weather = myutils.calculate_feels_like_tempeature(weather)
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
print('Done!')

print('Adding median reading for each building...')
if dataset == 'test':
    meter_train = pd.read_feather('./data/train.feather')
    meter_train_group = meter_train.groupby('building_id')['meter_reading']
    building_median = meter_train_group.median().astype(np.float16)
    del meter_train, meter_train_group
else:
    all_data_group = all_data.groupby('building_id')['meter_reading']
    building_median = all_data_group.median().astype(np.float16)
    del all_data_group
    
all_data['building_median'] = all_data['building_id'].map(building_median)
print('Done!')

if interpMissing:
    print('Interpolating missing data...')
    # interpolate missing values using a cubic spline
    nan_cols = all_data.loc[:, all_data.isnull().any()].columns
    all_data = myutils.replace_nan(all_data, nan_cols, 'interp')
    print('Done!')

print('Aligning timestamps..')
# Align timestamp offset
all_data = myutils.align_timestamp(all_data, offset)
print('Done!')

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
