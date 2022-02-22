#!/usr/bin/env python3
import time
import pickle
from itertools import product
from multiprocessing import Pool

import src.exports.data as data
import src.exports.state as state
import src.framework as framework
import src.output as output
import src.budgets as budgets
import src.fluxes as fluxes
import src.timescales as timescales
from src.unpacking import unpack_state_estimates
from src.ati import find_solution
from src.constants import LAYERS, GRID, ZG

def run_model(priors_from, gamma, rel_err):

    all_data = data.load_data()
    poc_data = data.process_poc_data(all_data['POC'])
    tracers = state.define_tracers(poc_data)
    params = state.define_params(all_data['NPP'], priors_from, rel_err)
    residuals = state.define_residuals(params['Po']['prior'], gamma)
    state_elements = framework.define_state_elements(tracers, params, LAYERS)
    equation_elements = framework.define_equation_elements(tracers, LAYERS)
    xo = framework.define_prior_vector(tracers, residuals, params, LAYERS)
    Co = framework.define_cov_matrix(tracers, residuals, params, LAYERS)

    ati_results = find_solution(
        tracers, state_elements, equation_elements, xo, Co, GRID, ZG, True)
    xhat, Ckp1, _, _ = ati_results
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, LAYERS)
    tracer_estimates, residual_estimates, param_estimates = estimates

    output.merge_by_keys(tracer_estimates, tracers)
    output.merge_by_keys(param_estimates, params)
    output.merge_by_keys(residual_estimates, residuals)

    residuals_sym = budgets.get_symbolic_residuals(residuals)
    residual_estimates_by_zone = budgets.integrate_by_zone(
        residuals_sym, state_elements, Ckp1, residuals=residuals)
    output.merge_by_keys(residual_estimates_by_zone, residuals)

    inventories_sym = budgets.get_symbolic_inventories(tracers)
    inventories = budgets.integrate_by_zone_and_layer(
        inventories_sym, state_elements, Ckp1, tracers=tracers)

    int_fluxes_sym = fluxes.get_symbolic_int_fluxes()
    int_fluxes = budgets.integrate_by_zone_and_layer(
        int_fluxes_sym, state_elements, Ckp1, tracers=tracers, params=params)

    residence_times = timescales.calculate_residence_times(
        inventories_sym, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
        tracers, params, state_elements, Ckp1)

    turnover_times = timescales.calculate_turnover_times(
        inventories_sym, int_fluxes_sym, int_fluxes, tracers, params,
        state_elements, Ckp1)

    to_pickle = (tracers, params, residuals, inventories, int_fluxes,
                 residence_times, turnover_times)
    
    save_path = f'../../results/exports/{priors_from}_{rel_err}_{gamma}.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)

if __name__ == '__main__':

    start_time = time.time()

    study_sites = ('NA', 'SP')
    gammas = (0.5, 1, 5, 10)
    rel_errs = (0.1, 0.2, 0.5, 1)

    n_cores = 8
    pool = Pool(n_cores)
    pool.starmap(run_model, product(study_sites, gammas, rel_errs))

    print(f'--- {(time.time() - start_time)/60} minutes ---')