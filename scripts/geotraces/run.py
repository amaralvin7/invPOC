#!/usr/bin/env python3
import time
import pickle
from numpy import diff
from itertools import product
from multiprocessing import Pool

import src.geotraces.data as data
import src.geotraces.state as state
import src.framework as framework
import src.output as output
from src.unpacking import unpack_state_estimates
from src.ati import find_solution

relative_err = 1
gamma = 0.15

poc_data = data.load_poc_data()
mixed_layer_depths = data.load_mixed_layer_depths()
ppz_data = data.load_ppz_data()

Lp_priors = data.get_Lp_priors(poc_data)
Po_priors = data.get_Po_priors(Lp_priors)
resid_prior_err = data.get_residual_prior_error(Po_priors, mixed_layer_depths)

priors_from = ('NA', 'SP')
stations = poc_data['station'].unique()

def invert_station(priors_from, station):

    mld = mixed_layer_depths[station]
    ppz = ppz_data[station]
    station_poc = data.get_station_poc(poc_data, station)
    tracers = state.define_tracers(station_poc)
    residuals = state.define_residuals(resid_prior_err, gamma)
    params = state.define_params(
        Lp_priors[station], Po_priors[station], priors_from, relative_err)

    grid = tuple(station_poc['depth'].values)
    layers = tuple(range(len(grid)))
    thick = diff((0,) + grid)

    state_elements = framework.define_state_elements(tracers, params, layers)
    equation_elements = framework.define_equation_elements(tracers, layers)
    xo = framework.define_prior_vector(tracers, residuals, params, layers)
    Co = framework.define_cov_matrix(tracers, residuals, params, layers)

    xhat, Ckp1, convergence_evolution, cost_evolution = find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, ppz, False,
        priors_from, station)
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, layers)
    tracer_estimates, residual_estimates, param_estimates = estimates

    output.merge_by_keys(tracer_estimates, tracers)
    output.merge_by_keys(param_estimates, params)
    output.merge_by_keys(residual_estimates, residuals)

    to_pickle = (tracers, params, residuals, grid, ppz, mld, layers,
                 convergence_evolution, cost_evolution)
    save_path = f'../../results/geotraces/stn{int(station)}_{priors_from}.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)

if __name__ == '__main__':

    start_time = time.time()

    n_cores = 32
    pool = Pool(n_cores)
    pool.starmap(invert_station, product(priors_from, stations))

    print(f'--- {(time.time() - start_time)/60} minutes ---')