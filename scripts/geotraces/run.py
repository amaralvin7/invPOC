import os
import pickle
import random
import time
import warnings
import sys

import itertools
import multiprocessing as mp
import numpy as np
import pandas as pd

import src.ati as ati
import src.geotraces.data as data
import src.framework as framework
import src.unpack as unpack

warnings.simplefilter('error')

poc_data = data.poc_by_station()
param_uniformity = data.define_param_uniformity()
npp_data = data.extract_nc_data(poc_data, 'cbpm')
Lp_priors = data.get_Lp_priors(poc_data)
ez_depths = data.get_ez_depths(Lp_priors)
Po_priors = data.get_Po_priors(poc_data, Lp_priors, npp_data)
B3_priors = data.get_B3_priors(npp_data)
station_data = data.get_station_data(poc_data, param_uniformity, ez_depths)

def generate_param_sets(n_runs):

    median_POCS = data.get_median_POCS()
    params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    compilation = pd.read_excel(
        '../../../geotraces/paramcompilation.xlsx', sheet_name=None)
    random.seed(0)
    
    extrema = get_param_extrema(compilation, params, median_POCS)
    
    param_sets = []
    for i in range(n_runs):
        param_set = {}
        param_set['id'] = i
        for p in params:
            param_set[p] = random.uniform(*extrema[p])
        param_sets.append(param_set)

    return param_sets

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
    
    q1, q3 = [np.percentile(values, p) for p in (25, 75)]
    iqr = q3 - q1
    lo_limit = q1 - (iqr * 1.5)
    hi_limit = q3 + (iqr * 1.5)
    inliers = [i for i in values if i >= lo_limit and i <= hi_limit]
    min_max = (min(inliers), max(inliers))
    
    return min_max

def invert_station(args):
    
    station, mc_params = args
    grid = station_data[station]['grid']
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

    try:  # if there are numerical instabilies in the ATI, return
        xhat, Ckp1, conv_ev, cost_ev, converged = ati.find_solution(
            equation_elements, xo, Co, grid, zg, umz_start,
            state_elements=state_elements)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            print(f'Singular matrix: {int(mc_params["id"])}, {station}')
            return
        else:
            raise
    except RuntimeWarning:
        print(f'RuntimeWarning: {int(mc_params["id"])}, {station}')
        return

    success = ati.success_check(converged, state_elements, xhat, Ckp1, zg)

    if success:

        x_resids = ati.normalized_state_residuals(xhat, xo, Co)
        tracer_estimates, param_estimates = unpack.unpack_state_estimates(
            tracers, params, state_elements, xhat, Ckp1, layers)

        unpack.merge_by_keys(tracer_estimates, tracers)
        unpack.merge_by_keys(param_estimates, params)
        
        results = {'tracers': tracers,
                   'params': params,
                   'x_resids': x_resids,
                   'convergence_evolution': conv_ev,
                   'cost_evolution': cost_ev}

        filename = f'ps{int(mc_params["id"])}_stn{station}.pkl'
        
        with open(os.path.join(save_path, filename), 'wb') as f:
            pickle.dump(results, f)
    

if __name__ == '__main__':
    
    start_time = time.time()
    
    n_param_sets = 125000
    
    save_path = f'../../results/geotraces/mc_{n_param_sets}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    param_sets = generate_param_sets(n_param_sets)

    stations = poc_data.keys()
    station_list = []
    set_list = []

    for s in param_sets:
        station_list.extend(stations)
        set_list.extend(itertools.repeat(s, len(stations)))

    inputs = zip(station_list, set_list)
    with mp.Pool(64, maxtasksperchild=1) as p:
        p.imap_unordered(invert_station, inputs)
        p.close()
        p.join()

    print(f'--- {(time.time() - start_time)/60} minutes ---')