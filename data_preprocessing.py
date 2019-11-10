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

dataset = 'train'

print('Loading data...')
if dataset == 'train':
    meter_data = pd.read_feather('./data/train.feather')
elif dataset == 'test':
    meter_data = pd.read_feather('./data/test.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')
weather_test = pd.read_feather('./data/weather_test.feather')
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

if dataset == 'train':
# Removing bad data in Site 0
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588
    print('Removing bad data from site 0...')
    meter_data = meter_data.loc[(meter_data['building_id'] > 104) | (meter_data['timestamp'] > '2016-05-20'), :]
    print('Done!')

print('Encoding building metadata and merging to meter data...')
# Encoding building metadata
building['primary_use'] = building['primary_use'].astype('category')
label = LabelEncoder()
building['primary_use'] = label.fit_transform(building['primary_use']).astype('uint8')
building['floor_count'] = building['floor_count'].fillna(0).astype('uint8')
building['year_built'] = building['year_built'].fillna(-999).astype('uint8')

# Merging building metadata and train/test sets
meter_data = meter_data.merge(building, on='building_id', how='left')
del building
print('Done!')

print('Adding new features to weather data...')
# Stuff done separetely
if dataset == 'train':
    weather = weather_train
    del weather_train
elif dataset == 'test':
    weather = weather_test
    del weather_test

weather = myutils.preprocess_datetime(weather, date_feat=['h', 'd', 'w', 'm', 'wk'])
weather = myutils.encode_cyclic_feature(weather, 'weekday', 7)
weather = myutils.encode_cyclic_feature(weather, 'hour', 24)
weather = myutils.encode_cyclic_feature(weather, 'day', 31)
weather = myutils.encode_cyclic_feature(weather, 'month', 12)
print('Done!')

print('Merging weather and meter dataframes...')
# Merge weather data
all_data = meter_data.merge(weather, on=['site_id', 'timestamp'], how='left')
all_data = myutils.reduce_mem_usage(all_data)
del weather
print('Done!')

print('Interpolating missing data...')
# interpolate missing values using a cubic spline
nan_cols = all_data.loc[all_data.isna().any(axis=1), :].columns
for col in nan_cols:
    all_data[col] = all_data[col].fillna(all_data[col].interpolate(method='polynomial', order=3))
all_data = myutils.reduce_mem_usage(all_data)
print('Done!')

print('Aligning timestamps..')
# Align timestamp offset
all_data = myutils.align_timestamp(all_data, offset)
print('Done!')

print('Saving data to feather...')
if dataset == 'train':
    all_data.to_feather('./data/train_clean.feather')
elif dataset == 'test':
    all_data.to_feather('./data/test_clean.feather')
print('Done!')

del all_data
gc.collect()
