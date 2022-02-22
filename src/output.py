#!/usr/bin/env python3

def merge_by_keys(merge_this, into_this):
    
    for i in into_this:
        for j in merge_this[i]:
            into_this[i][j] = merge_this[i][j]
            