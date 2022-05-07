#!/usr/bin/env python3
import time
import pickle
from numpy import diff
from itertools import product
from multiprocessing import Pool
import sys
import src.geotraces.data as data
import src.geotraces.state as state
import src.framework as framework
import src.output as output
import src.budgets as budgets
import src.fluxes as fluxes
import src.timescales as timescales
from src.unpacking import unpack_state_estimates
from src.ati import find_solution

relative_err = 1
gamma = 0.2
maxdepth = 600

poc_data = data.load_poc_data()
ppz_data = data.load_ppz_data()
npp_data = data.load_npp_data()
mixed_layer_depths = data.load_mixed_layer_depths()

Lp_priors = data.get_Lp_priors(poc_data)
Po_priors = data.get_Po_priors(npp_data, Lp_priors)
B3_priors = data.get_B3_priors(npp_data)
resid_prior_err = data.get_residual_prior_error(Po_priors, mixed_layer_depths)

priors_from_tuple = ('NA', 'SP')
priors_from_tuple = ('NA',)
stations = poc_data['station'].unique()

def invert_station(priors_from, station):

    mld = mixed_layer_depths[station]
    ppz = ppz_data[station]
    station_poc = data.get_station_poc(poc_data, station, maxdepth)
    tracers = state.define_tracers(station_poc)
    residuals = state.define_residuals(resid_prior_err, gamma)
    params = state.define_params(
        Lp_priors[station], Po_priors[station], B3_priors[station],
        priors_from, relative_err)

    grid = tuple(station_poc['depth'].values)
    layers = tuple(range(len(grid)))
    zg = min(grid, key=lambda x:abs(x - ppz))  # grazing depth
    umz_start = grid.index(zg) + 1
    zone_layers = ('EZ', 'UMZ') + layers
    thick = diff((0,) + grid)

    state_elements = framework.define_state_elements(tracers, params, layers)
    equation_elements = framework.define_equation_elements(tracers, layers)
    xo = framework.define_prior_vector(tracers, residuals, params, layers)
    Co = framework.define_cov_matrix(tracers, residuals, params, layers)

    xhat, Ckp1, convergence_evolution, cost_evolution = find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, zg, mld,
        False, umz_start, priors_from, station)

    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, layers)
    tracer_estimates, residual_estimates, param_estimates = estimates

    output.merge_by_keys(tracer_estimates, tracers)
    output.merge_by_keys(param_estimates, params)
    output.merge_by_keys(residual_estimates, residuals)

    umz_start = grid.index(zg) + 1

    residuals_sym = budgets.get_symbolic_residuals(
        residuals, umz_start, layers)
    residual_estimates_by_zone = budgets.integrate_by_zone(
        residuals_sym, state_elements, Ckp1, residuals=residuals)
    output.merge_by_keys(residual_estimates_by_zone, residuals)

    inventories_sym = budgets.get_symbolic_inventories(
        tracers, umz_start, layers, thick)
    inventories = budgets.integrate_by_zone_and_layer(
        inventories_sym, state_elements, Ckp1, layers, tracers=tracers)

    int_fluxes_sym = fluxes.get_symbolic_int_fluxes(
        umz_start, layers, thick, grid, mld, zg)
    int_fluxes = budgets.integrate_by_zone_and_layer(
        int_fluxes_sym, state_elements, Ckp1, layers, tracers=tracers,
        params=params)

    sink_fluxes = fluxes.sinking_fluxes(
        layers, state_elements, Ckp1, tracers, params)

    residence_times = timescales.calculate_residence_times(
        inventories_sym, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
        tracers, params, state_elements, Ckp1, zone_layers, umz_start)

    turnover_times = timescales.calculate_turnover_times(
        inventories_sym, int_fluxes_sym, int_fluxes, tracers, params,
        state_elements, Ckp1, zone_layers)

    to_pickle = (tracers, params, residuals, inventories, int_fluxes,
                 sink_fluxes, residence_times, turnover_times, grid, zg, mld,
                 layers, convergence_evolution, cost_evolution)
    save_path = f'../../results/geotraces/stn{int(station)}_{priors_from}.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)

if __name__ == '__main__':

    start_time = time.time()

    n_processes = 32

    pool = Pool(32)
    pool.starmap(invert_station, product(priors_from_tuple, stations))

    print(f'--- {(time.time() - start_time)/60} minutes ---')