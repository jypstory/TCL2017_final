#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: lee ji su
"""


import os
import scipy as sp
from glob import glob


train_bad_path_src = "./data/train/bad_img"
train_good_path_src = "./data/train/good_img"
test_bad_path_src = "./data/test/bad_img"
test_good_path_src = "./data/test/good_img"


train_bad_list = sorted(glob(train_bad_path_src + '/' + '*.png'))
train_bad_data = tuple((1, sp.misc.imresize(sp.misc.imread(bad), size=(100, 100))) for bad in train_bad_list)
print('TRAIN Bad Data : %s' % len(train_bad_data))

train_good_list = sorted(glob(train_good_path_src + '/' + '*.png'))
train_good_data = tuple((0, sp.misc.imresize(sp.misc.imread(good), size=(100, 100))) for good in train_good_list)
print('TRAIN Good Data : %s' % len(train_good_data))

# data = train_bad_data + train_good_data
# train, test = model_selection.train_test_split(data, train_size=.7, test_size=.3, random_state=11)

test_bad_list = sorted(glob(test_bad_path_src + '/' + '*.png'))
test_bad_data = tuple((1, sp.misc.imresize(sp.misc.imread(bad), size=(100, 100))) for bad in test_bad_list)
print('TEST Bad Data : %s' % len(test_bad_data))

test_good_list = sorted(glob(test_good_path_src + '/' + '*.png'))
test_good_data = tuple((0, sp.misc.imresize(sp.misc.imread(good), size=(100, 100))) for good in test_good_list)
print('TEST Good Data : %s' % len(test_good_data))

train = train_bad_data + train_good_data
print('TRAIN Bad & Good : %s' % len(train))

test = test_bad_data + test_good_data
print('TEST Bad & Good : %s' % len(test))