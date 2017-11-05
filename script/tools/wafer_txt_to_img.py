#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: lee ji su
"""

import os
from os import path
import numpy as np
from PIL import Image, ImageOps

path_src = "../../wafer_data/pattern_txt/arc"
path_out = "../../wafer_data/denoised_pattern_image"
abspath_src = path.abspath(path_src)
abspath_out = path.abspath(path_out)

## to delete .Ds_store file
def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]
print(mylistdir(path_src))

file_paths = [path.join(abspath_src, name) for name in mylistdir(path_src) if path.isfile(path.join(abspath_src, name))]

for file in file_paths:
    data = np.genfromtxt(file, delimiter=';', dtype=None)
    coordinates = data[data[:,4]==b'W'][:,2:4].astype(np.uint8)
    array = np.zeros(shape=(50,70), dtype=np.uint8)
    array[coordinates[:,1], coordinates[:,0]] = 1

    wafer = array.astype(np.float64)
    for __ in range(0,49):
        for _ in range(0,69):
            wafer[__,_] = (1/11)*array[__-1,__-1] + (1/11)*array[__-1,_] + (1/11)*array[__-1,_+1] + (1/11)*array[__,_-1] + (3/11)*array[__,_] + (1/11)*array[__,_+1] + (1/11)*array[__+1,_-1] + (1/11)*array[__+1,_] + (1/11)*array[__+1,_+1]

            if wafer[__,_] > 0.0001:
                wafer[__,_] = 255
            else :
                wafer[__,_] = 0

    wafer = wafer.astype(np.uint8)
    img = Image.fromarray(wafer).crop((10, 10, 69, 49)).resize((100,100))
    img = ImageOps.invert(img)
    lot_id = file.split(path.sep)[-1].split('.')[0]
    img_filename = abspath_out + '/' + lot_id + '.png'
    print(img_filename)
    img.save(img_filename)
    print(lot_id, '-- image generated')