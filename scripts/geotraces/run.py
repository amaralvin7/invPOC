#!/usr/bin/env python3
import time
import pickle
from numpy import diff
from itertools import product

import src.geotraces.data as data
import src.geotraces.state as state
import src.framework as framework
import src.output as output
from src.unpacking import unpack_state_estimates
from src.ati import find_solution

start_time = time.time()

poc_data = data.load_poc_data()
mixed_layer_depths = data.load_mixed_layer_depths()
ppz_data = data.load_ppz_data()

Lp_prior = data.get_Lp_priors(poc_data)
P30_prior = data.get_Po_priors(Lp_prior)

super_stations = (8.0, 14.0, 23.0, 29.0, 35.0, 39.0)
priors_from = ('NA', 'SP')

for s, p in product(super_stations, priors_from):

    mld = mixed_layer_depths[s]
    ppz = ppz_data[s]
    station_poc = data.get_station_poc(poc_data, s)
    tracers = state.define_tracers(station_poc)
    residuals = state.define_residuals(P30_prior[s], mld)
    params = state.define_params(Lp_prior[s], P30_prior[s], p)

    grid = tuple(station_poc['depth'].values)
    layers = tuple(range(len(grid)))
    thick = diff((0,) + grid)

    state_elements = framework.define_state_elements(tracers, params, layers)
    equation_elements = framework.define_equation_elements(tracers, layers)
    xo = framework.define_prior_vector(tracers, residuals, params, layers)
    Co = framework.define_cov_matrix(tracers, residuals, params, layers)

    xhat, Ckp1, convergence_evolution, cost_evolution = find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, ppz, False)
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, layers)
    tracer_estimates, residual_estimates, param_estimates = estimates

    output.merge_by_keys(tracer_estimates, tracers)
    output.merge_by_keys(param_estimates, params)
    output.merge_by_keys(residual_estimates, residuals)

    to_pickle = (tracers, params, residuals, grid, ppz, mld, layers)
    save_path = f'../../results/geotraces/stn{int(s)}_{p}.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)

print(f'--- {(time.time() - start_time)/60} minutes ---')