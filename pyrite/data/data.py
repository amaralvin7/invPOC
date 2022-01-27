#!/usr/bin/env python3
import pandas as pd
import numpy as np
from constants import MLD, GRID, GAMMA
from os import path

def load_data():
    
    module_path = path.abspath(__file__)
    pkg_path = path.dirname(module_path)
    data_file_path = path.join(pkg_path,'data.xlsx')

    return pd.read_excel(data_file_path, sheet_name=None)

def process_poc_data(to_process):

    processed = pd.DataFrame(GRID, columns=['depth'])
    processed['n_casts'] = [
        get_number_of_casts(to_process, depth) for depth in GRID]

    for tracer in ('POCS', 'POCL'):
        mean, sd = calculate_mean_and_sd(to_process, tracer)
        processed[tracer] = mean
        processed[f'{tracer}_se'] = (sd  / np.sqrt(processed['n_casts']))
        
    return processed

def get_number_of_casts(to_process, depth):

    return len(to_process[to_process['mod_depth'] == depth])

def calculate_mean_and_sd(to_process, tracer):

    mean, sd = [], []

    for depth in GRID:
        at_depth = to_process[to_process['mod_depth'] == depth][tracer]
        mean.append(at_depth.mean())
        sd.append(at_depth.std())

    relative_sd_50m = sd[1]/mean[1]  # 50m is the second grid depth
    sd[0] = mean[0] * relative_sd_50m  # 30m is the first grid depth

    return mean, sd

def define_tracers(data):
    
    tracers = {'POCS': {}, 'POCL': {}}
    
    for t in tracers:
        tracers[t]['prior'] = data[t]
        tracers[t]['prior_e'] = data[f'{t}_se']
    
    return tracers

def define_residuals(proportional_to):
    
    residuals = {'POCS': {}, 'POCL': {}}
    
    for tracer in residuals:
        residuals[tracer]['prior'] = 0
        residuals[tracer]['prior_e'] = GAMMA * proportional_to * MLD
    
    return residuals
    