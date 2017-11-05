#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: lee ji su
"""

import os
from os import path
import numpy as np
from PIL import Image, ImageOps, ImageDraw
import math
import random

##### TODO #####
# 1. Add 'Center Shift' to draw_circle(inner circle)
# 2. Add 'Varying Thickness' to draw_line
# 3. Add Other Patterns (Like Defocus, Dounut, Radial Lines ...)
# 4. Add a fuction to draw Arc Patterns(Not a Circle)
# 5. Enhance Noise Pattern

# To write files
relpath = "../../wafer_data/bad_image/line"
abspath = path.abspath(relpath)

# Constants
BLACK = 0
WHITE = 255
die_area = (8, 9, 64, 43)
arr_dim = (die_area[3]-die_area[1], die_area[2]-die_area[0])
img_dim = 100

# --------------------------------
# Make Circle using a Radius
# --------------------------------
def draw_circle(array, center, radius=140, weight=10, color='black'):
    img = Image.new(mode='1', size=(img_dim, img_dim), color='white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([(center[0]-radius, center[1]-radius),
                  (center[0]+radius, center[1]+radius)],
                 outline=color, fill=color) # outer
    draw.ellipse([(center[0]-radius+weight, center[1]-radius+weight),
                  (center[0]+radius-weight, center[1]+radius-weight)],
                 outline=color, fill='white') # innor
    del draw
    mask_arr = np.array(img.resize((arr_dim[1], arr_dim[0])).getdata(), np.uint8)\
                 .reshape(arr_dim)
    mask = mask_arr == BLACK
    array[mask] = BLACK
    return array

# --------------------------------
# Draw Straight Line
# --------------------------------
def draw_line(array, position, weight=10, color='black'):
    img = Image.new(mode='1', size=(img_dim, img_dim), color='white')
    draw = ImageDraw.Draw(img)
    draw.line(position, fill=color, width=weight)
    del draw
    mask_arr = np.array(img.resize((arr_dim[1], arr_dim[0])).getdata(), np.uint8)\
                 .reshape(arr_dim)
    mask = mask_arr == BLACK
    array[mask] = BLACK
    return array

# --------------------------------
# Add Noise
# --------------------------------
def add_noise(array, count, color=BLACK):
    mask = np.zeros(shape=arr_dim[0]*arr_dim[1], dtype=np.bool)
    mask[:count] = True
    np.random.shuffle(mask)
    mask = mask.reshape(arr_dim)
    array[mask] = color
    return array

# --------------------------------
# Make Wafer Map Mask
# --------------------------------
def adjust_mask(array):
    mask = np.zeros((arr_dim[0], arr_dim[1]), dtype=bool)
    mask[11-die_area[1]+0,24:-24] = True
    mask[11-die_area[1]+1,18:-18] = True
    mask[11-die_area[1]+2,15:-15] = True
    mask[11-die_area[1]+3,13:-13] = True
    mask[11-die_area[1]+4,11:-11] = True
    mask[11-die_area[1]+5,9:-9] = True
    mask[11-die_area[1]+6,8:-8] = True
    mask[11-die_area[1]+7,7:-7] = True
    mask[11-die_area[1]+8,6:-6] = True
    mask[11-die_area[1]+9,5:-5] = True
    mask[11-die_area[1]+10,4:-4] = True
    mask[11-die_area[1]+11,4:-4] = True
    mask[11-die_area[1]+12,3:-3] = True
    mask[11-die_area[1]+13,3:-3] = True
    mask[11-die_area[1]+14,3:-3] = True
    mask[11-die_area[1]+15,3:-3] = True
    mask[11-die_area[1]+16,3:-3] = True
    mask[11-die_area[1]+17,3:-3] = True
    mask[11-die_area[1]+18,4:-4] = True
    mask[11-die_area[1]+19,4:-4] = True
    mask[11-die_area[1]+20,5:-5] = True
    mask[11-die_area[1]+21,5:-5] = True
    mask[11-die_area[1]+22,6:-6] = True
    mask[11-die_area[1]+23,7:-7] = True
    mask[11-die_area[1]+24,9:-9] = True
    mask[11-die_area[1]+25,10:-10] = True
    mask[11-die_area[1]+26,12:-12] = True
    mask[11-die_area[1]+27,14:-14] = True
    mask[11-die_area[1]+28,17:-17] = True
    mask[11-die_area[1]+29,21:-21] = True
    array[~mask] = WHITE
    return array

# --------------------------------
# Draw Outline (arc + line)
# --------------------------------
def draw_outline(img):
    wf_dim = int(img_dim * 7/15)
    offset_y = int(img_dim * 1/200)
    offset_cut = int(img_dim * 9/20)
    x = math.sqrt(wf_dim**2 - offset_cut**2)
    theta = math.atan(x / offset_cut) * 180 / math.pi

    draw = ImageDraw.Draw(img)
    draw.arc([(img_dim/2-wf_dim, img_dim/2-wf_dim+offset_y),
              (img_dim/2+wf_dim, img_dim/2+wf_dim+offset_y)],
             start=90.+theta, end=450.-theta, fill='black')
    draw.line([(img_dim/2-x, img_dim/2+offset_cut+offset_y),
               (img_dim/2+x, img_dim/2+offset_cut+offset_y)],
              fill='black', width=1)
    del draw
    return img

def make_random_wafer_img(filename, arc_para, line_para, noise_para, outline=True):
    # 1. Set Parameters
    arc_num = arc_para[0]
    arc_para= arc_para[1]

    line_num = line_para[0]
    line_para = line_para[1]

    noise_black_num = noise_para[0]
    noise_white_num = noise_para[1]

    # 2. Make an Empty Wafer as array
    array = np.full(shape=arr_dim, fill_value=WHITE, dtype=np.uint8)

    # 3. Draw Random Arcs
    for i in range(0, arc_num):
        array = draw_circle(array, center=arc_para[i][0],
                                   radius=arc_para[i][1],
                                   weight=arc_para[i][2])

    # 4. Draw Random Lines
    for i in range(0, line_num):
        array = draw_line(array, position=line_para[i][0],
                                 weight=line_para[i][1])

    # 5. Add Noise
    array = add_noise(array, noise_black_num, color=BLACK)
    array = add_noise(array, noise_white_num, color=WHITE)

    # 6. Adjust Mask
    array = adjust_mask(array)

    # 7. Resize
    img = Image.fromarray(array).resize((img_dim, img_dim)) # resizing

    # 8. Draw Outline
    if outline:
        img = draw_outline(img)

    # 9. Save Image
    #img.show()
    img.save(abspath + path.sep + filename + '.png')
    print(filename + ' -- image generated')


if __name__ == '__main__':
    # --------------------------------
    # Set Parameters
    # --------------------------------
    for i in range(1, 100):  ## craete 1000 image
        filename = str(i)

        arc_num = 0
        arc_para = []
        # ((center.x, center.y), radius, stroke_weight)
        arc_para.append(((0, 0), 70, 3))
        arc_para.append(((100, 50), 70, 5))
        arc_para = (arc_num, arc_para)

        line_num = 1
        line_para = []
        # (position([start.x, start.y), (end.x, end.y)]), stroke_weight)
        start_x = random.randint(1, 10)
        start_y = random.randint(2, 5)
        end_x = random.randint(90, 100)
        end_y = random.randint(300, 350)

        if (math.sqrt((start_x - end_x) ** 2 + (start_y - end_y) ** 2)) < 30: continue

        line_para.append(([(start_x, start_y), (end_x, end_y)], 3))
        line_para = (line_num, line_para)

        noise_black_num = 150  # num of black noise
        noise_white_num = 20  # num of black noise
        noise_para = (noise_black_num, noise_white_num)

        # --------------------------------
        # Make Wafer Image
        # --------------------------------
        make_random_wafer_img(filename, arc_para, line_para, noise_para, outline=False)