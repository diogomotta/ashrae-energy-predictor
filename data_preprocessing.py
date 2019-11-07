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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# My modules
import myutils
from lgbm_predictor import lgbm_model

train = pd.read_feather('./data/train.feather')
test = pd.read_feather('./data/test.feather')
building = pd.read_feather('./data/building_metadata.feather')
weather_train = pd.read_feather('./data/weather_train.feather')
weather_test = pd.read_feather('./data/weather_test.feather')
    
# Removing bad data in Site 0
# https://www.kaggle.com/c/ashrae-energy-prediction/discussion/113054#656588
train = train.loc[(train['building_id'] > 104) | (train['timestamp'] > '2016-05-20'), :]

# Encoding building metadata
building['primary_use'] = building['primary_use'].astype('category')
label = LabelEncoder()
building['primary_use'] = label.fit_transform(building['primary_use']).astype(np.int8)
building['floor_count'] = building['floor_count'].fillna(0)
building['year_built'] = building['year_built'].fillna(-999)

# Merging building metadata and train/test sets
train = train.merge(building, on='building_id', how='left')
test = test.merge(building, on='building_id', how='left')
del building

# Align timestamps
weather = pd.concat([weather_train, weather_test],ignore_index=True)
offset = myutils.calculate_time_offset(weather)
weather_train = myutils.align_timestamp(weather_train, offset)
weather_test = myutils.align_timestamp(weather_test, offset)
del weather

# Time features
weather_train = myutils.preprocess_datetime(weather_train, date_feat=['h', 'w', 'm', 'dw'])

# Merge weather data
train = train.loc[train['meter']==0, :]
train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
del weather_train, weather_test
gc.collect()

print('Train dataframe')
train = myutils.reduce_mem_usage(train)
#################################################################################
# Holdout set 1
# Split train set by building_id -> 20% to houldout
train_build, oof_build = train_test_split(train['building_id'].unique(), test_size=0.20)
oof_1 = train[train['building_id'].isin(oof_build)].reset_index(drop=True)
train = train[train['building_id'].isin(train_build)].reset_index(drop=True)

sns.countplot(x='site_id', data=train)
sns.countplot(x='site_id', data=oof_1)

drop_cols = ['meter_reading', 'timestamp', 'meter']
feat_cols = [col for col in list(train) if col not in drop_cols]
X = train[feat_cols]
y = np.log1p(train['meter_reading'])
model = lgbm_model(X, y)

print(myutils.rmse(model.predict(oof_1[feat_cols]), np.log1p(oof_1['meter_reading'])))

#del train, test, weather_train, weather_test  
gc.collect()