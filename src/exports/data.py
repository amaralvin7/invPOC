import pandas as pd
import numpy as np
from os import path

import sys
def load_data():
    
    module_path = path.abspath(__file__)
    src_parent_path = module_path.split('src')[0]
    data_file_path = path.join(src_parent_path,'data/exports.xlsx')
    
    return pd.read_excel(data_file_path, sheet_name=None)

def process_poc_data(to_process, grid):

    processed = pd.DataFrame(grid, columns=['depth'])
    processed['n_casts'] = [
        get_number_of_casts(to_process, depth) for depth in grid]

    for tracer in ('POCS', 'POCL'):
        mean, sd = calculate_mean_and_sd(to_process, tracer, grid)
        processed[tracer] = mean
        processed[f'{tracer}_se'] = (sd  / np.sqrt(processed['n_casts']))
        
    return processed

def get_number_of_casts(to_process, depth):

    return len(to_process[to_process['mod_depth'] == depth])

def calculate_mean_and_sd(to_process, tracer, grid):

    mean, sd = [], []

    for depth in grid:
        at_depth = to_process[to_process['mod_depth'] == depth][tracer]
        mean.append(at_depth.mean())
        sd.append(at_depth.std())

    relative_sd_50m = sd[1]/mean[1]  # 50m is the second grid depth
    sd[0] = mean[0] * relative_sd_50m  # 30m is the first grid depth

    return mean, sd
    