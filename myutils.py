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
import datetime
from sklearn.metrics import mean_squared_error

''' Kernels
https://www.kaggle.com/corochann/ashrae-training-lgbm-by-meter-type
https://www.kaggle.com/frednavruzov/aligning-temperature-timestamp
https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction
https://www.kaggle.com/kyakovlev/ashrae-cv-options
https://www.kaggle.com/kyakovlev/ashrae-data-minification
https://www.kaggle.com/kyakovlev/ashrae-baseline-lgbm
https://www.kaggle.com/c/ashrae-energy-prediction/discussion/116773
https://www.kaggle.com/rohanrao/ashrae-divide-and-conquer
https://www.kaggle.com/purist1024/ashrae-simple-data-cleanup-lb-1-08-no-leaks
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
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def convert_float16_to_float32(df, verbose=True):
    numerics = ['float16']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            df[col] = df[col].astype(np.float32)  
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
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
    

def align_timestamp(df, offset, strategy='14h_calc', site_id=None, tz=None):
    if strategy == '14h_calc':
        df['offset'] = df['site_id'].map(offset)
        df['timestamp_aligned'] = (df['timestamp'] - pd.to_timedelta(df['offset'], unit='H'))
        df['timestamp'] = df['timestamp_aligned']
        del df['timestamp_aligned'], df['offset']
    elif strategy == 'timezone':
        df.loc[df['site_id']==site_id, 'timestamp'] = df['timestamp'].dt.tz_localize('utc').dt.tz_convert(tz).dt.tz_localize(None)
        
    return df

def preprocess_datetime(df, date_feat=['h', 'd', 'w', 'm', 'wk']):
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
        elif feat == 'wk':
            df['weekend'] = np.where((df['timestamp'].dt.weekday == 5) | (df['timestamp'].dt.weekday == 6), 1, 0)
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
			df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.mean()))
		elif strategy == 'median_group':
			df[col] = df.groupby(group)[col].transform(lambda x: x.fillna(x.median()))
		elif strategy == 'mode_group':
			df[col] = df.groupby(group)[col].apply(lambda x: x.fillna(x.mode()[0])) 
		elif strategy == 'interp':
			df[col] = df[col].fillna(df[col].interpolate(method='polynomial', order=3))
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
    
def encode_cyclic_feature(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    return df

def calculate_feels_like_tempeature(weather_df):
    weather_df['feels_like_temperature'] = 0
    
    # Heat index for high temperature and high humidity
    hi_rows = weather_df.loc[(weather_df['air_temperature'] > 27) & (weather_df['relative_humidity'] > 40)].index
    weather_df.loc[hi_rows, 'feels_like_temperature'] = calculate_heat_index(weather_df.iloc[hi_rows])
    
    # Wind chill for cold temperature and windy weather
    wc_rows = weather_df.loc[(weather_df['air_temperature'] <= 10) & (weather_df['wind_speed'] > 1.34)].index
    weather_df.loc[wc_rows, 'feels_like_temperature'] = calculate_wind_chill(weather_df.iloc[wc_rows])
    
    # Regular temperature otherwise
    other_rows = weather_df.loc[weather_df['feels_like_temperature'] == 0].index
    weather_df.loc[other_rows, 'feels_like_temperature'] = weather_df.loc[other_rows, 'air_temperature']
    
    return weather_df
    
def calculate_wind_chill(weather_df):
    T = weather_df['air_temperature'] * (9/5) + 32
    V = weather_df['wind_speed'] * (1609.34 / 3600)
    wind_chill = 35.74 + (0.6215 * T) - 35.75 * V**0.16 + 0.4275 * T * V**0.16
    
    return ((wind_chill - 32) * (5/9))

def calculate_heat_index(weather_df):
    c1 = -42.379
    c2 = 2.04901523
    c3 = 10.14333127
    c4 = -0.22475541
    c5 = -6.83783e-3
    c6 = -5.481717e-2
    c7 = 1.22874e-3
    c8 = 8.5282e-4
    c9 = -1.99e-6

    T = weather_df['air_temperature'] * (9/5) + 32
    RH = weather_df['relative_humidity']

    heat_index = (  c1 +
                    c2 * T +
                    c3 * RH +
                    c4 * T * RH +
                    c5 * (T**2) +
                    c6 * (RH**2) +
                    c7 * (T**2) * RH +
                    c8 * T * (RH**2) +
                    c9 * (T**2) * (RH**2)
                    )

    return ((heat_index - 32) * (5/9))

def calculate_relative_humidity(weather_df):
    T = weather_df['air_temperature']
    TD = weather_df['dew_temperature']
    b = 17.625
    c = 243.04
    
    weather_df['relative_humidity'] = 100 * np.exp(TD * b / (TD + c) - b * T / (c + T))
    weather_df['relative_humidity'].clip(lower=0, upper=100, inplace=True)
    
    return weather_df

def create_lag_features(df, cols, group, window, calc):
    """
    Creating lag-based features looking back in time.
    """
    
    grouped = df.groupby(group)
    
    rolled = grouped[cols].rolling(window=window, min_periods=0)
    
    if calc == 'mean':
        df_calc = rolled.mean().reset_index().astype(np.float32)
    elif calc == 'median':
        df_calc = rolled.median().reset_index().astype(np.float32)
    elif calc == 'min':
        df_calc = rolled.min().reset_index().astype(np.float32)
    elif calc == 'max':
        df_calc = rolled.max().reset_index().astype(np.float32)
    elif calc == 'std':
        df_calc = rolled.std().reset_index().astype(np.float32)
    
    for col in cols:
        label = '{:}_{:}_lag{:d}'.format(col, calc, window)
        df[label] = df_calc[col] 
        
    return df

def find_missing_dates(weather_df):
    # Find Missing Dates
    time_format = "%Y-%m-%d %H:%M:%S"
    start_date = weather_df['timestamp'].min()
    end_date = weather_df['timestamp'].max()
    total_hours = int(((end_date - start_date).total_seconds() + 3600) / 3600)
    hours_list = np.asarray([(end_date - datetime.timedelta(hours=x)).strftime(time_format) for x in range(total_hours)], dtype='datetime64')

    for site_id in weather_df['site_id'].unique():
        site_hours = np.array(weather_df[weather_df['site_id'] == site_id]['timestamp'])
        new_rows = pd.DataFrame(np.setdiff1d(hours_list,site_hours),  columns=['timestamp'])
        new_rows['site_id'] = site_id
        print('Adding {:d} new rows to site_id {:d}'.format(len(new_rows), site_id))
        weather_df = pd.concat([weather_df, new_rows], sort=True)

        weather_df = weather_df.reset_index(drop=True)           
        
    return weather_df

def replace_with_leaked(original, leaked):
    merged = original.merge(leaked, on=['building_id', 'meter', 'timestamp'], how='left')
    merged['meter_reading'] = 0
    leaked_ok_len = len(merged.loc[merged['meter_reading_y'].isnull()==False, 'meter_reading_y'])
    print('Replacing {:d} rows with leaked data'.format(leaked_ok_len))
    merged.loc[merged['meter_reading_y'].isnull()==False, 'meter_reading'] = merged.loc[merged['meter_reading_y'].isnull()==False, 'meter_reading_y']
    original_len = len(merged.loc[merged['meter_reading_y'].isnull()==True, 'meter_reading_x'])
    print('Keeping {:d} rows with original data'.format(original_len))
    merged.loc[merged['meter_reading_y'].isnull()==True, 'meter_reading'] = merged.loc[merged['meter_reading_y'].isnull()==True, 'meter_reading_x']
    merged.drop(columns=['meter_reading_x', 'meter_reading_y'], inplace=True)
    
    return merged

def make_is_bad_zero(df, min_interval=48, summer_start=6, summer_end=8):
    """Helper routine for 'find_bad_zeros'.
    
    This operates upon a single dataframe produced by 'groupby'. We expect an 
    additional column 'meter_id' which is a duplicate of 'meter' because groupby 
    eliminates the original one."""
    meter = df['meter_id'].iloc[0]
    is_zero = df['meter_reading'] == 0
    if meter == 0:
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero

    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()
    ids = all_sequence_ids[is_zero].rename("ids")
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        keep = set(ids[(df['timestamp'].dt.month <= summer_start) |
                       (df['timestamp'].dt.month >= summer_end)].unique())
        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)
    elif meter == 1:
        time_ids = ids.to_frame().join(df['timestamp']).set_index("timestamp").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result

def find_bad_zeros(X, y):
    """Returns an Index object containing only the rows which should be deleted."""
    Xy = X.assign(meter_reading=y, meter_id=X.meter)
    is_bad_zero = Xy.groupby(["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])

def find_bad_rows(X, y):
    return find_bad_zeros(X, y)

def calculate_gradient(w0, h, y_pred, y_leak):
    
    # Calculate current function value
    X_comb = np.array([w * y for w, y in zip(w0, y_pred)]) / np.sum(w0) # normalized X vector
    f_comb = np.sqrt(mean_squared_error(np.log1p(X_comb), np.log1p(y_leak)))
    
    # Calculate the partial derivatives
    grad = list()
    for i in range(0, len(w0)):
        w1 = w0
        w1[i] += h
        y_comb_ = np.array([w * y for w, y in zip(w1, y_pred)]) / np.sum(w1)
        f_comb_ = np.sqrt(mean_squared_error(np.log1p(y_comb_), np.log1p(y_leak)))
        grad.append( (f_comb_ - f_comb) / h)
        
    return np.asarray(grad)
        
def gradient_descent(w0, y_pred, y_leak, gamma, max_iters, precision):
    w_next = w0
    
    for _i in range(max_iters):
        w_curr = w_next
        w_next = w_curr - gamma * calculate_gradient(w0, 1e-4, y_pred, y_leak)

        step = w_next - w_curr
        if abs(step) <= precision:
            break

    print('Minimum at ', w_next)
        
    