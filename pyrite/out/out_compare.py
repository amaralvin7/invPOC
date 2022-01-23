#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:51:02 2020

@author: vamaral

compare if two output files are the same
True == match
"""

import filecmp
print(filecmp.cmp('pyrite_out_dvmTrue_NA_test.txt', 'out.txt', shallow=False))