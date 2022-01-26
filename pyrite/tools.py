#!/usr/bin/env python3

def slice_by_species(to_slice, species, state_elements):

    sliced = [to_slice[i] for i, e in enumerate(
        state_elements) if e.split('_')[0] == species]
    
    return sliced

def merge_by_keys(merge_this, into_this):
    
    for i in into_this:
        for j in merge_this[i]:
            into_this[i][j] = merge_this[i][j]