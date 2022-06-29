#!/usr/bin/env python3
import random
import time

from itertools import repeat
from multiprocessing import Pool
import numpy as np
import pandas as pd
import pickle

import src.ati as ati
import src.geotraces.data as data
import src.geotraces.state as state
import src.framework as framework
import src.budgets as budgets
from src.unpacking import unpack_state_estimates


gamma = 0.1

poc_data = data.poc_by_station()
stations = poc_data.keys()
mixed_layer_depths = data.load_mixed_layer_depths()

npp_data = data.extract_nc_data(poc_data, 'cbpm')
Lp_priors = state.get_Lp_priors(poc_data)
ez_depths = state.get_ez_depths(Lp_priors)
Po_priors = state.get_Po_priors(poc_data, Lp_priors, npp_data, ez_depths)
B3_priors = state.get_B3_priors(npp_data)
resid_prior_err = state.get_residual_prior_error(
    poc_data, Lp_priors, npp_data, ez_depths)

def monte_carlo_table(n_runs):

    median_POCS = data.get_median_POCS()
    mc_params = ('B2p', 'Bm2', 'Bm1s', 'Bm1l', 'ws', 'wl')
    compil = pd.read_excel('../../../geotraces/paramcompilation.xlsx', sheet_name=None)
    random.seed(0)
    
    extrema = {}
    for p in mc_params:
        if p == 'B2p':
            lo, hi = param_range(compil['B2']['val'].to_numpy())
            lo = lo / median_POCS
            hi = hi / median_POCS
        else:
            lo, hi = param_range(compil[p]['val'].to_numpy())
        extrema[p] = (lo, hi)
    
    rows = []
    for i in range(n_runs):
        row = {}
        row['id'] = i
        for p in mc_params:
            row[p] = random.uniform(*extrema[p])
        rows.append(row)

    df = pd.DataFrame(rows)

    return df

def param_range(values):
    
    q1, q3 = [np.percentile(values, p) for p in (25, 75)]
    iqr = q3 - q1
    lo_limit = q1 - (iqr * 1.5)
    hi_limit = q3 + (iqr * 1.5)
    inliers = [i for i in values if i >= lo_limit and i <= hi_limit]
    min_max = (min(inliers), max(inliers))
    
    return min_max

def invert_station(station, mc_params):

    mld = mixed_layer_depths[station]
    tracers = state.define_tracers(poc_data[station])
    residuals = state.define_residuals(resid_prior_err, gamma)
    params = state.define_params(
        Lp_priors[station], Po_priors[station], B3_priors[station],
        mc_params)

    grid = tuple(poc_data[station]['depth'].values)
    layers = tuple(range(len(grid)))
    zg = min(grid, key=lambda x:abs(x - ez_depths[station]))  # grazing depth
    umz_start = grid.index(zg) + 1
    # zone_layers = ('EZ', 'UMZ') + layers
    # thick = diff((0,) + grid)

    state_elements = framework.define_state_elements(tracers, params, layers)
    equation_elements = framework.define_equation_elements(tracers, layers)
    xo = framework.define_prior_vector(tracers, residuals, params, layers)
    Co = framework.define_cov_matrix(tracers, residuals, params, layers)

    xhat, Ckp1, convergence_evolution, cost_evolution, converged = ati.find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, zg,
        umz_start, mld)

    nonnegative = ati.nonnegative_check(state_elements, xhat)
    success = converged and nonnegative

    return converged, nonnegative, success
    
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, layers)
    tracer_estimates, residual_estimates, param_estimates = estimates

    tools.merge_by_keys(tracer_estimates, tracers)
    tools.merge_by_keys(param_estimates, params)
    tools.merge_by_keys(residual_estimates, residuals)

    umz_start = grid.index(zg) + 1

    residuals_sym = budgets.get_symbolic_residuals(
        residuals, umz_start, layers)
    residual_estimates_by_zone = budgets.integrate_by_zone(
        residuals_sym, state_elements, Ckp1, residuals=residuals)
    tools.merge_by_keys(residual_estimates_by_zone, residuals)
    
    to_pickle = (tracers, params, residuals, grid, zg, mld,
                 layers, convergence_evolution, cost_evolution)
    
    mcid = int(mc_params['id'])
    save_path = f'../../results/geotraces/{mcid}_stn{int(station)}.pkl'
    with open(save_path, 'wb') as file:
        pickle.dump(to_pickle, file)

if __name__ == '__main__':

    start_time = time.time()
    
    n_runs = 7000
    mc_table = monte_carlo_table(n_runs)
    convergence = []
    nonnegative = []
    success = []
    
    n_processes = 22
    for _, row in mc_table.iterrows():
        i = int(row['id'])
        print(i)
        pool = Pool(n_processes)
        result = pool.starmap(invert_station, zip(stations, repeat(row)))
        convergence.append(all([r[0] for r in result]))
        nonnegative.append(all([r[1] for r in result]))
        success.append(all([r[2] for r in result]))

        # with open(f'../../results/geotraces/to_delete/{i}.pkl', 'wb') as file:
        #     pickle.dump(i, file)
        # for s in stations:
        #     invert_station(s, row)
    
    mc_table['convergence'] = convergence
    mc_table['nonnegative'] = nonnegative
    mc_table['success'] = success
    mc_table.to_csv('../../results/geotraces/table.csv', index=False)

    print(f'--- {(time.time() - start_time)/60} minutes ---')