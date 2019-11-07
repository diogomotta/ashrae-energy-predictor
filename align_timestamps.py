#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 11:24:08 2019

@author: diogo
"""
import pandas as pd
import matplotlib.pyplot as plt

weather_train_df = pd.read_csv('./data/weather_train.csv', parse_dates=['timestamp'])
weather_test_df = pd.read_csv('./data/weather_test.csv', parse_dates=['timestamp'])


def calculate_offset(df_weather):
    weather_key = ['site_id', 'timestamp']
    
    temp_skeleton = weather[weather_key + ['air_temperature']].drop_duplicates(subset=weather_key).sort_values(by=weather_key).copy()
    
    # calculate ranks of hourly temperatures within date/site_id chunks
    temp_skeleton['temp_rank'] = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.date])['air_temperature'].rank('average')
    
    # create a dataframe of site_ids (0-16) x mean hour rank of temperature within day (0-23)
    df_2d = temp_skeleton.groupby(['site_id', temp_skeleton.timestamp.dt.hour])['temp_rank'].mean().unstack(level=1)
    
    # Subtract the columnID of temperature peak by 14, getting the timestamp alignment gap.
    offset = pd.Series(df_2d.values.argmax(axis=1) - 14)
    offset.index.name = 'site_id'
    
    return offset
    

def timestamp_align(df, offset):
    df['offset'] = df.site_id.map(offset)
    df['timestamp_aligned'] = (df.timestamp - pd.to_timedelta(df.offset, unit='H'))
    df['timestamp'] = df['timestamp_aligned']
    del df['timestamp_aligned']
    return df

weather = pd.concat([weather_train_df, weather_test_df],ignore_index=True)

offset = calculate_offset(weather)
weather_train_df = timestamp_align(weather_train_df, offset)
weather_test_df = timestamp_align(weather_test_df, offset)