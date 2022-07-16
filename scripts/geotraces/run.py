import os
from pickle import dump
from random import seed, uniform
from time import time

from itertools import repeat
from multiprocessing import Pool
from numpy import percentile
from pandas import DataFrame, read_excel
from tqdm import tqdm

import src.ati as ati
import src.geotraces.data as data
import src.framework as framework
from src.unpacking import unpack_state_estimates, merge_by_keys


poc_data = data.poc_by_station()
param_uniformity = data.define_param_uniformity()
npp_data = data.extract_nc_data(poc_data, 'cbpm')
Lp_priors = data.get_Lp_priors(poc_data)
ez_depths = data.get_ez_depths(Lp_priors)
Po_priors = data.get_Po_priors(poc_data, Lp_priors, npp_data, ez_depths)
B3_priors = data.get_B3_priors(npp_data)
station_data = data.get_station_data(poc_data, param_uniformity, ez_depths)

def generate_param_sets(n_runs):

    median_POCS = data.get_median_POCS()
    params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    compilation = read_excel(
        '../../../geotraces/paramcompilation.xlsx', sheet_name=None)
    seed(0)
    
    extrema = get_param_extrema(compilation, params, median_POCS)
    
    rows = []
    for i in range(n_runs):
        row = {}
        row['id'] = i
        for p in params:
            row[p] = uniform(*extrema[p])
        rows.append(row)

    return DataFrame(rows)

def get_param_extrema(compilation, params, median_POCS):

    extrema = {}
    for p in params:
        if p == 'B2p':
            lo, hi = get_param_range(compilation['B2']['val'].to_numpy())
            lo = lo / median_POCS
            hi = hi / median_POCS
        else:
            lo, hi = get_param_range(compilation[p]['val'].to_numpy())
        extrema[p] = (lo, hi)
    
    return extrema

def get_param_range(values):
    
    q1, q3 = [percentile(values, p) for p in (25, 75)]
    iqr = q3 - q1
    lo_limit = q1 - (iqr * 1.5)
    hi_limit = q3 + (iqr * 1.5)
    inliers = [i for i in values if i >= lo_limit and i <= hi_limit]
    min_max = (min(inliers), max(inliers))
    
    return min_max

def invert_station(station, mc_params):

    grid = station_data[station]['grid']
    mld = station_data[station]['mld']
    layers = station_data[station]['layers']
    zg = station_data[station]['zg']
    umz_start = station_data[station]['umz_start']
    state_elements = station_data[station]['s_elements']
    equation_elements = station_data[station]['e_elements']
    tracers = station_data[station]['tracers'].copy()
    params = param_uniformity.copy()

    data.set_param_priors(params, Lp_priors[station], Po_priors[station],
                          B3_priors[station], mc_params)

    xo = framework.define_prior_vector(tracers, params, layers)
    Co = framework.define_cov_matrix(tracers, params, layers)

    xhat, Ckp1, convergence_evolution, cost_evolution, converged = ati.find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, zg,
        umz_start, mld)

    nonnegative = ati.nonnegative_check(state_elements, xhat, Ckp1)
    success = converged and nonnegative

    if success:

        x_resids = ati.normalized_state_residuals(xhat, xo, Co)
        tracer_estimates, param_estimates = unpack_state_estimates(
            tracers, params, state_elements, xhat, Ckp1, layers)

        merge_by_keys(tracer_estimates, tracers)
        merge_by_keys(param_estimates, params)
        
        results = {'tracers': tracers,
                   'params': params,
                   'x_resids': x_resids,
                   'convergence_evolution': convergence_evolution,
                   'cost_evolution': cost_evolution}

        filename = f'ps{int(mc_params["id"])}_stn{station}.pkl'
        
        with open(os.path.join(save_path, filename), 'wb') as f:
            dump(results, f)
            
        return station
    

if __name__ == '__main__':
    
    start_time = time()
    
    save_path = '../../results/geotraces/mc_hard_25k_uniform_iqr_bug'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    n_runs = 25000
    mc_table = generate_param_sets(n_runs)

    stations = poc_data.keys()
    n_processes = len(stations)
    set_successes = []
    station_successes = []
    
    for _, row in tqdm(mc_table.iterrows(), total=len(mc_table)):
        i = int(row['id'])
        pool = Pool(n_processes)
        successes = pool.starmap(invert_station, zip(stations, repeat(row)))
        successes = [x for x in successes if x is not None]
        station_successes.append(successes)
        set_successes.append(len(stations) == len(successes))

    mc_table['set_success'] = set_successes
    mc_table['station_successes'] = station_successes
    with open(os.path.join(save_path, 'table.pkl'), 'wb') as f:
        dump(mc_table, f)

    print(f'--- {(time() - start_time)/60} minutes ---')