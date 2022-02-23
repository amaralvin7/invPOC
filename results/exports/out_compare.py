#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 15:51:02 2020

@author: vamaral

compare if two output files are the same
True == match
"""
f1 = 'test_full.txt'
f2 = 'out.txt'

import filecmp
print(filecmp.cmp(f1, f2, shallow=False))

# import difflib
# with open(f1) as file_1:
#     file_1_text = file_1.readlines()
  
# with open(f2) as file_2:
#     file_2_text = file_2.readlines()
  
# # Find and print the diff:
# for line in difflib.unified_diff(
#         file_1_text, file_2_text, fromfile=f1, 
#         tofile=f2, lineterm=''):
#     print(line)