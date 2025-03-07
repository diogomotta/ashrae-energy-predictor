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
cleanBadRows = True
cleanBadRows_meters123 = True
cleanKaggleRows = False

# Leaked data
useMyLeak = False
useKaggleLeak = False

print('Loading data...')
if dataset == 'train':
    meter_train = pd.read_feather('./data/train.feather')
elif dataset == 'test':
    meter_data = pd.read_feather('./data/test.feather')
    meter_train = pd.read_feather('./data/train.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')
weather_test = pd.read_feather('./data/weather_test.feather')
if cleanKaggleRows:
    rowsToDrop = pd.read_csv('./data/rows_to_drop.csv')
    meter_train = meter_train.drop(rowsToDrop['0'])
    meter_train.reset_index(drop=True, inplace=True)
print('Done!')

if useKaggleLeak:
    leak = pd.read_feather('./data/kaggle_leak.feather')
    leak.fillna(0, inplace=True)
    leak = leak[(leak['timestamp'].dt.year == 2016)]
    leak.loc[leak['meter_reading'] < 0, 'meter_reading'] = 0 # remove large negative values
    leak = leak[leak['building_id'] != 245]
    print('Replacing original data with leaked data...')
    meter_train = myutils.replace_with_leaked(meter_train, leak)
    print('Done!')
    
    print('Including data from 2017 and 2018...')
    leak = pd.read_feather('./data/kaggle_leak.feather')
    leak.fillna(0, inplace=True)
    leak = leak[(leak['timestamp'].dt.year > 2016)]
    meter_train = meter_train.append(leak, ignore_index=True, sort=True)
    
    del leak
    gc.collect()

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
if dataset == 'train':
    meter_train = meter_train.merge(building, on='building_id', how='left')
else:
    meter_train = meter_train.merge(building, on='building_id', how='left')
    meter_data = meter_data.merge(building, on='building_id', how='left')
del building
print('Done!')

if cleanKaggleRows is False:
    # Removing bad data in Site 0
    # https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588
    print('Removing bad data from site 0...')
    meter_train = meter_train.loc[(meter_train['building_id'] > 104) | (meter_train['timestamp'] > '2016-05-20'), :]
    print('Done!')
    
if cleanBadRows:
    # Remove bad rows using code from
    # https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
    print('Removing null meter 0 readings...')
    bad_rows = meter_train.loc[(meter_train['meter_reading']==0) &
                               (meter_train['meter']==0), :].index
    meter_train = meter_train.drop(index=bad_rows)
    print('Done!')

if cleanKaggleRows is False:
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
    
print('Converting meter 0 data in site 0 from kBTU to kWh...')
rows_idx = meter_train.loc[(meter_train['meter']==0) & (meter_train['site_id']==0), :].index
meter_train.loc[rows_idx, 'meter_reading'] = meter_train.loc[rows_idx, 'meter_reading'] * 0.2931
print('Done!')

if dataset == 'train':
    meter_data = meter_train
    del meter_train
    
print('Calculating mean and std building reading per meter...')
if dataset == 'test':
    meter_train['hour'] = meter_train['timestamp'].dt.hour
    meter_train_group = meter_train.groupby(['building_id', 'meter', 'hour'])['meter_reading']
    del meter_train
else:
    meter_data['hour'] = meter_data['timestamp'].dt.hour
    meter_train_group = meter_data.groupby(['building_id', 'meter', 'hour'])['meter_reading']
    del meter_data['hour']

building_median = np.log1p(meter_train_group.median()).astype(np.float32)
building_mean = np.log1p(meter_train_group.mean()).astype(np.float32)
building_std = np.log1p(meter_train_group.std()).astype(np.float32)
del meter_train_group
gc.collect()
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
    
gc.collect()

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
del weather['date']

# Replace remaining NaNs with daily mean value
nan_cols = weather.loc[:, weather.isnull().any()].columns
weather = myutils.replace_nan(weather, nan_cols, 'mean')

# Add relative humidity and feels like temperature
weather = myutils.calculate_relative_humidity(weather)
weather = myutils.calculate_feels_like_tempeature(weather)
if addLag:
    lag_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr',
                'sea_level_pressure', 'wind_direction', 'wind_speed']
    weather = myutils.create_lag_features(weather, lag_cols, 'site_id', 72, 'mean')
print('Done!')

if alignTimestamps:
    print('Aligning weather timestamps..')  
    # Using 14h calculation
    weather = myutils.align_timestamp(weather, offset, strategy='14h_calc')
    print('Done!')

print('Saving data to feather...')
if dataset == 'train':
    weather.to_feather('./data/weather_train_feats.feather')
elif dataset == 'test':
    weather.to_feather('./data/weather_test_feats.feather')
print('Done!')
  
print('Merging weather and meter dataframes...')
# Merge weather data
all_data = myutils.reduce_mem_usage(weather)
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

print('Adding mean and std building reading per meter...')
all_data['building_median'] = all_data[['building_id', 'meter', 'hour']].merge(building_median, on=['building_id', 'meter', 'hour'], how='left')['meter_reading']
all_data['building_mean'] = all_data[['building_id', 'meter', 'hour']].merge(building_mean, on=['building_id', 'meter', 'hour'], how='left')['meter_reading']
all_data['building_std'] = all_data[['building_id', 'meter', 'hour']].merge(building_std, on=['building_id', 'meter', 'hour'], how='left')['meter_reading']
print('Done!')

if dataset == 'train':
    print('Dropping remaining rows with NaNs...')
    # Drop remaining NaNs
    all_data = all_data.dropna()
    all_data.reset_index(drop=True, inplace=True)
    print('Done!')
else: 
    print('Replacing remaining NaNs with mean value...')
    # Replace remaining NaNs with daily mean value
    nan_cols = all_data.loc[:, all_data.isnull().any()].columns
    all_data = myutils.replace_nan(all_data, nan_cols, 'mean')
    print('Done!')

if interpMissing:
    print('Interpolating missing data...')
    # interpolate missing values using a cubic spline
    nan_cols = all_data.loc[:, all_data.isnull().any()].columns
    all_data = myutils.replace_nan(all_data, nan_cols, 'interp')
    print('Done!')

if dataset == 'train':
    if cleanBadRows_meters123:
        print('Removing null meter 1 readings when temperatures are higher than 10C...')
        bad_rows = myutils.remove_sequential_zero_rows(all_data.loc[all_data['meter']==1, :], low_temp_thresh=10)
        total_rows = len(all_data.loc[all_data['meter']==1, :])
        print('Dropping {:d} out of {:d} rows...'.format(len(bad_rows), total_rows))
        all_data = all_data.drop(index=bad_rows)
        print('Done!')
        
        print('Removing null meter 2 readings when temperatures are lower than 25C...')
        bad_rows = myutils.remove_sequential_zero_rows(all_data.loc[all_data['meter']==2, :], high_temp_thresh=25)
        total_rows = len(all_data.loc[all_data['meter']==2, :])
        print('Dropping {:d} out of {:d} rows...'.format(len(bad_rows), total_rows))
        all_data = all_data.drop(index=bad_rows)
        print('Done!')
        
        print('Removing null meter 3 readings when temperatures are higher than 25C...')
        bad_rows = myutils.remove_sequential_zero_rows(all_data.loc[all_data['meter']==3, :], high_temp_thresh=25)
        total_rows = len(all_data.loc[all_data['meter']==3, :])
        print('Dropping {:d} out of {:d} rows...'.format(len(bad_rows), total_rows))
        all_data = all_data.drop(index=bad_rows)
        print('Done!')
        all_data.reset_index(drop=True, inplace=True)
    
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
