"""Processing pipeline for EXPORTS POC concentration data."""
from pickle import load

import pandas as pd
import numpy as np

from src.exports.constants import GRID


def load_data():
    """Load all data required for inversions."""
    with open('../../data/exports/data.pkl', 'rb') as f:
        data = load(f)

    return data


def process_poc_data(to_process):
    """Process POC data to be used in inversions.

    Args:
        to_process (pd.DataFrame): Raw concentration data for each sample
        collected at every station.

    Returns:
        processed (pd.DataFrame): Contains mean and standard error of POC
        concentrations at each model grid depth, as well as the number of casts
        (i.e., samples) considered at each depth.
    """
    processed = pd.DataFrame(GRID, columns=['depth'])
    processed['n_casts'] = [
        get_number_of_casts(to_process, depth) for depth in GRID]

    for tracer in ('POCS', 'POCL'):
        mean, sd = calculate_mean_and_sd(to_process, tracer)
        processed[tracer] = mean
        processed[f'{tracer}_se'] = (sd / np.sqrt(processed['n_casts']))

    return processed


def get_number_of_casts(to_process, depth):
    """Get the number of casts considered at a given depth."""
    n_casts = len(to_process[to_process['mod_depth'] == depth])

    return n_casts


def calculate_mean_and_sd(to_process, tracer):
    """Calculate the mean and standard deviation at each depth."""
    mean, sd = [], []

    for depth in GRID:
        at_depth = to_process[to_process['mod_depth'] == depth][tracer]
        mean.append(at_depth.mean())
        sd.append(at_depth.std())

    relative_sd_50m = sd[1] / mean[1]  # 50m is the second GRID depth
    sd[0] = mean[0] * relative_sd_50m  # 30m is the first GRID depth

    return mean, sd
