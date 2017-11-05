#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:59:40 2017
@author: pydemia
"""

from pydemia.manufacture.new_lot import Wafer

# %% Line
line_lot = Wafer(fac_id='Factory1',
                 lot_cd='ABC',
                 end_tm='20171101123456',
                 size=(60, 40),
                 unit_cnt=10,
                 f_type='line',
                 y_val=.95, # yield
                 pattern=['#', 'W'])

line_lot.unitplot(unit_id='01', pattern=['W'], figsize=(6, 5), c=['red'])

line_lot.to_txt('./wafer_data/pattern_txt/line', sep=';')


# %% Bold Line
bold_line_lot = Wafer(fac_id='Factory1',
                      lot_cd='ABC',
                      end_tm='20171101123456',
                      size=(60, 40),
                      unit_cnt=10,
                      f_type='bold_line',
                      y_val=.91,
                      pattern=['#', 'W'])

bold_line_lot.unitplot(unit_id='01', pattern=['W'], figsize=(6,5), c=['red'])

bold_line_lot.to_txt('./wafer_data/pattern_txt//bold_line', sep=';')


# %% Arc
arc_lot = Wafer(fac_id='Factory1',
                lot_cd='ABC',
                end_tm='20171101123456',
                size=(60, 40),
                unit_cnt=10,
                f_type='arc',
                y_val=.95,
                pattern=['#', 'W'])

arc_lot.unitplot(unit_id='01', pattern=['W'], figsize=(6,5), c=['red'])

arc_lot.to_txt('./wafer_data/pattern_txt//arc', sep=';')


# %% None
normal_lot = Wafer(fac_id='Factory1',
                   lot_cd='ABC',
                   end_tm='20171101123456',
                   size=(60, 40),
                   unit_cnt=10,
                   f_type='none',
                   y_val=.95,
                   pattern=['#', 'W'])

normal_lot.unitplot(unit_id='01', pattern=['W'], figsize=(6,5), c=['red'])

normal_lot.to_txt('./wafer_data/pattern_txt//normal', sep=';')