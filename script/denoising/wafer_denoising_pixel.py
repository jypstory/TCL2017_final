#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: lee ji su
"""


import os
from os import path
import scipy.misc as sp_misc
from PIL import Image, ImageOps
from glob import glob


path_src = "../../wafer_data/bad_image/gan_line"
path_out = "../../wafer_data/bad_image/denoise_line"
abspath_src = path.abspath(path_src)
abspath_out = path.abspath(path_out)

file_paths = [path.join(abspath_src, name) for name in os.listdir(path_src) if path.isfile(path.join(abspath_src, name))]

bad_list = sorted(glob(path_src + '/' + '*.png'))
bad_data = list((sp_misc.imresize(sp_misc.imread(bad), size=(100,100))) for bad in bad_list)
noise = 10

for idx, val in enumerate(bad_data):
    val[val < noise] = 0
    val[val >= noise] = 255
    val = 255 - val
    #img = Image.fromarray(val).resize((80, 80)).resize((100, 100))
    img = Image.fromarray(val).resize((100, 100))
    img = ImageOps.invert(img)
    img_filename = abspath_out + '//_' + str(idx) + '.png'
    img.save(img_filename)