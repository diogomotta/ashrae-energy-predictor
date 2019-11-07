# -*- coding: utf-8 -*-
"""
Spyder Editor

Este é um arquivo de script temporário.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.close('all')
import seaborn as sns
import gc
import plotly.express as px
from plotly.offline import plot

# My modules
import myutils

train = pd.read_feather('./data/train.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')

# Merging building metadata and train/test sets
train = train.merge(building, on='building_id', how='left')
del building

# Time features
weather_train = myutils.preprocess_datetime(weather_train, date_feat=['h', 'w', 'm', 'dw'])

# Align timestamps
offset = myutils.calculate_time_offset(weather_train)
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
train = myutils.align_timestamp(train, offset)
train = myutils.reduce_mem_usage(train)
del weather_train
gc.collect()

# Air temperature histogram
plot_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
             'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
             'wind_speed']
for col in plot_cols:
    myutils.plot_dist_col(col, train)
    
# daily weather data
#train_daily = train.loc[train['building_id'].between(0, 100), ['timestamp', 'building_id', 'meter', 'meter_reading']]
train_daily = train.loc[:, ['timestamp', 'building_id', 'meter', 'meter_reading']]
train_daily['date'] = train_daily['timestamp'].dt.date
train_daily = train_daily.groupby(['date', 'building_id', 'meter']).sum()
train_daily = train_daily.reset_index()

fig = px.area(train_daily.loc[train_daily['meter']==0], x='date', y='meter_reading', color='building_id')
plot(fig)

train_daily_agg = train_daily.groupby(['date', 'meter']).agg(['sum', 'mean', 'idxmax', 'max'])
train_daily_agg = train_daily_agg.reset_index()
level_0 = train_daily_agg.columns.droplevel(0)
level_1 = train_daily_agg.columns.droplevel(1)
level_0 = ['' if x == '' else '-' + x for x in level_0]
train_daily_agg.columns = level_1 + level_0
train_daily_agg.rename_axis(None, axis=1)
train_daily_agg.head()

fig_total = px.line(train_daily_agg, x='date', y= 'meter_reading-sum', color='meter', render_mode='svg')
fig_total.update_layout(title='Total kWh per energy aspect')
plot(fig_total)
