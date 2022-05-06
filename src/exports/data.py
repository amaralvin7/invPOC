import pandas as pd
import numpy as np
from os import path

from src.exports.constants import GRID

import sys

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

    relative_sd_50m = sd[1]/mean[1]  # 50m is the second GRID depth
    sd[0] = mean[0] * relative_sd_50m  # 30m is the first GRID depth

    return mean, sd
    