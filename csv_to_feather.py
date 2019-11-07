#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:39:02 2019

@author: diogo
"""
import pandas as pd

# load data
train = pd.read_csv('./data/train.csv', parse_dates=['timestamp'])
test = pd.read_csv('./data/test.csv', parse_dates=['timestamp'])
building = pd.read_csv('./data/building_metadata.csv')
weather_train = pd.read_csv('./data/weather_train.csv', parse_dates=['timestamp'])
weather_test = pd.read_csv('./data/weather_test.csv', parse_dates=['timestamp'])

train.to_feather('train.feather')
test.to_feather('test.feather')
weather_train.to_feather('weather_train.feather')
weather_test.to_feather('weather_test.feather')
building.to_feather('building_metadata.feather')
