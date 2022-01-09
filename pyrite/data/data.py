#!/usr/bin/env python3
import pandas as pd
from constants import MLD, GRID
from os import path

def load_data():
    
    module_path = path.abspath(__file__)
    pkg_path = path.dirname(module_path)
    data_file_path = path.join(pkg_path,'data.xlsx')

    return pd.read_excel(data_file_path, sheet_name=None)

def process_poc_data(to_process):

    processed = pd.DataFrame(GRID, columns=['depth'])
    processed['ncasts'] = [
        get_number_of_casts(to_process, depth) for depth in GRID]

    for tracer in ('POCS', 'POCL'):
        mean, sd = calculate_mean_and_sd(to_process, tracer)
        processed[tracer] = mean
        processed[f'{tracer}_sd'] = sd
        # processed[f'{tracer}_se'] = calculate_se(to_process, tracer)
        
    return processed

def get_number_of_casts(to_process, depth):

    return len(to_process[to_process['mod_depth'] == depth])

def calculate_mean_and_sd(to_process, tracer):

    mean, sd = [], []

    for depth in GRID:
        at_depth = to_process[to_process['mod_depth'] == depth][tracer]
        mean.append(at_depth.mean())
        sd.append(at_depth.std())

    return mean, sd