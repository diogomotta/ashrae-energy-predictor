#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:21:15 2019

@author: diogo
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

''' Kernels
https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type
https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp
https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
https://www.kaggle.com/kyakovlev/ashrae-cv-options
https://www.kaggle.com/kyakovlev/ashrae-data-minification
https://www.kaggle.com/kyakovlev/ashrae-baseline-lgbm
'''

## Function to reduce the memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def calculate_time_offset(df_weather):
    weather_key = ['site_id', 'timestamp']
    
    temp_skeleton = df_weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
    
    # calculate ranks of hourly temperatures within date/site_id chunks
    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton['timestamp'].dt.date])['air_temperature'].rank('average')
    
    # create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton['timestamp'].dt.hour])['temp_rank'].mean().unstack(level=1)
    
    # Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
    offset = pd.Series(df_2d.values.argmax(axis=1) - 14)
    offset.index.name = 'site_id'
    return offset
    

def align_timestamp(df, offset):
    df['offset'] = df.site_id.map(offset)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['offset'], df['timestamp_aligned']
    return df

def preprocess_datetime(df, date_feat=['h', 'd', 'w', 'm']):
    # Incude datetime features from timestamp
    for feat in date_feat:
        if feat == 'h':
            df['hour'] = df['timestamp'].dt.hour.astype(np.int8)
        elif feat == 'd':
            df['day'] = df['timestamp'].dt.day
        elif feat == 'w':
            df['weekday'] = df['timestamp'].dt.weekday
        elif feat == 'm':
            df['month'] = df['timestamp'].dt.month
    return df

def replace_nan(df, cols, strategy, value=None, group=None):
	for col in cols:
		if strategy == 'mean':
			val = df.loc[df[col].isnull() == False, col].mean()
			df[col].fillna(val, inplace=True)
		elif strategy == 'median':
			val = df.loc[df[col].isnull() == False, col].median()
			df[col].fillna(val, inplace=True)
		elif strategy == 'mode':
			val = df.loc[df[col].isnull() == False, col].mode()[0]
			df[col].fillna(val, inplace=True)
		elif strategy == 'const':
			val = value
			df[col].fillna(val, inplace=True)
		elif strategy == 'mean_group':
			df[col] = df.groupby([group])[col].transform(lambda x: x.fillna(x.mean()))
		elif strategy == 'median_group':
			df[col] = df.groupby([group])[col].transform(lambda x: x.fillna(x.median()))
		elif strategy == 'mode_group':
			df[col] = df.groupby([group])[col].apply(lambda x: x.fillna(x.mode()[0])) 
	return df
            
def merge_dataframes(df, target_meter, weather_df):
    target_df = df.loc[df['meter']==target_meter, :]
    target_df = target_df.merge(weather_df, on=['site_id', 'timestamp'], how='left')
    df = pd.get_dummies(df)
    return df

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def plot_dist_col(column, train, test=None):
    '''plot dist curves for train and test weather data for the given column name'''
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.distplot(train[column].dropna(), color='green', label='train', ax=ax).set_title(column, fontsize=16)
    if test is not None:
        sns.distplot(test[column].dropna(), color='purple', label='test', ax=ax).set_title(column, fontsize=16)
    plt.xlabel(column, fontsize=16)
    plt.legend()
    plt.show()