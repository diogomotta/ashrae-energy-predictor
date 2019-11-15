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
from scipy.interpolate import CubicSpline

# My modules
import myutils

meter_data = pd.read_feather('./data/train.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')

# plot control
plotFig = True
plotWeatherHist = False
plotArea = False

# Processing control
doInterp = False

# Time features
weather_train = myutils.preprocess_datetime(weather_train, date_feat=['h', 'w', 'm', 'd'])

# Show data from buildings with mean steam reading above 5k
temp_df = meter_data.groupby(['building_id', 'meter']).mean()
temp_df = temp_df.reset_index()
build_id = temp_df.loc[(temp_df['meter']==2) & (temp_df['meter_reading'] > 5e3), 'building_id']

# Show data from those buildings
plot_df = meter_data.loc[(meter_data['building_id'].isin(build_id)) & (meter_data['meter']==2), :]
fig_total = px.line(plot_df, x='timestamp', y='meter_reading', color='building_id', render_mode='svg')
fig_total.update_layout(title='Total kWh for Steam')
plot(fig_total)
del plot_df

plot_df = meter_data.loc[(meter_data['building_id'].isin(build_id)) & (meter_data['meter']==0), :]
fig_total = px.line(plot_df, x='timestamp', y='meter_reading', color='building_id', render_mode='svg')
fig_total.update_layout(title='Total kWh for Electricity')
plot(fig_total)
del plot_df

# Removing bad data in building 1099 on meter 2 (steam)
print('Removing bad data from building 1099...')
rows_idx = meter_data.loc[(meter_data['building_id'] == 1099) & (meter_data['meter']==2), :].index
meter_data.loc[rows_idx, 'meter_reading'] = meter_data.loc[rows_idx, 'meter_reading'] / 1e3

plt.rcParams['figure.figsize'] = (18, 10)
temp_df = meter_data.loc[meter_data['meter']==0, 'meter_reading'].apply(np.log1p)
ax = sns.kdeplot(temp_df, shade = True, label='electricity')
temp_df = meter_data.loc[meter_data['meter']==1, 'meter_reading'].apply(np.log1p)
ax = sns.kdeplot(temp_df, shade = True, label='chilled water', color = 'm')
temp_df = meter_data.loc[meter_data['meter']==2, 'meter_reading'].apply(np.log1p)
ax = sns.kdeplot(temp_df, shade = True, label='steam', color = 'lime')
temp_df = meter_data.loc[meter_data['meter']==3, 'meter_reading'].apply(np.log1p)
ax = sns.kdeplot(temp_df, shade = True, label='hot water', color = 'k')
ax.set_xlabel('Log(Meter Reading)', fontsize = 20)
del temp_df

if doInterp:
    # View missing data and interpolation results
    # Find month with most missing data for each column
    nan_cols = ['timestamp', 'site_id', 
                'air_temperature', 'cloud_coverage', 'dew_temperature',
                'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    weather_monthly = weather_train.loc[:, nan_cols]
    weather_monthly['month'] = weather_monthly['timestamp'].dt.month
    weather_monthly = weather_monthly.groupby(['month', 'site_id']).count()
    nan_cols = ['sea_level_pressure']
    for col in nan_cols:
        min_cnt = weather_monthly[col].min()
        if min_cnt == 0:
            min_cnt = np.unique(weather_monthly[col])[1]
        month_min = weather_monthly.loc[weather_monthly[col]==min_cnt, :].index[0][0]
        site_id_min = weather_monthly.loc[weather_monthly[col]==min_cnt, :].index[0][1]
        
        time_plt = pd.to_datetime(weather_train.loc[(weather_train['month']==month_min) &
                                            (weather_train['site_id']==site_id_min) &
                                            (weather_train[col].isnull()==False), 'timestamp'])
        col_plt = weather_train.loc[(weather_train['month']==month_min) &
                                            (weather_train['site_id']==site_id_min) &
                                            (weather_train[col].isnull()==False), col]
        cs = CubicSpline(time_plt, col_plt)
        time_interp = pd.date_range(time_plt.min(), time_plt.max(), freq='H')
        col_interp = cs(time_interp)
        
        if plotFig:
            plt.figure()
            plt.plot(time_plt, col_plt, 'o-')
            plt.plot(time_interp, col_interp, '.-')
            plt.title('{:} interpolation'.format(col))
    del weather_monthly

# Merging building metadata and train/test sets
meter_data = meter_data.merge(building, on='building_id', how='left')
del building

# Align timestamps
offset = myutils.calculate_time_offset(weather_train)
meter_data = meter_data.merge(weather_train, on=['site_id', 'timestamp'], how='left')
meter_data = myutils.align_timestamp(meter_data, offset)
meter_data = myutils.reduce_mem_usage(meter_data)
del weather_train
gc.collect()

if plotWeatherHist:
    # Air temperature histogram
    plot_cols = ['air_temperature', 'cloud_coverage', 'dew_temperature',
                 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction',
                 'wind_speed']
    for col in plot_cols:
        myutils.plot_dist_col(col, meter_data)
    
# daily weather data
#train_daily = train.loc[train['building_id'].between(0, 100), ['timestamp', 'building_id', 'meter', 'meter_reading']]
train_daily = meter_data.loc[:, ['timestamp', 'building_id', 'meter', 'meter_reading']]
train_daily['date'] = train_daily['timestamp'].dt.date
train_daily = train_daily.groupby(['date', 'building_id', 'meter']).sum()
train_daily = train_daily.reset_index()

if plotArea:
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

if plotFig:
    fig_total = px.line(train_daily_agg, x='date', y= 'meter_reading-sum', color='meter', render_mode='svg')
    fig_total.update_layout(title='Total kWh per energy aspect')
    plot(fig_total)