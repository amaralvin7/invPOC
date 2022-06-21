"""Invert EXPORTS POC concentration data."""
from itertools import product
from multiprocessing import Pool
from pickle import dump
from time import time

import src.ati as ati
import src.budgets as budgets
import src.exports.data as data
import src.exports.state as state
import src.fluxes as fluxes
import src.framework as framework
import src.timescales as timescales
from src.exports.constants import *
from src.modelequations import calculate_B2
from src.unpacking import unpack_state_estimates


def run_model(priors_from, gamma, rel_err):
    """Run the inverse model and pickle results.

    Args:
        priors_from (str): Location from which to pick B2p and Bm2 priors. Can
        be NA (North Atlantic) or SP (Station P).
        gamma (float): Proportionality constant for setting model error.
        rel_err (float): Relative error for prior estimates for which error
        must be set arbitrarily.
    """
    all_data = data.load_data()
    poc_data = data.process_poc_data(all_data['POC'])
    tracers = state.define_tracers(poc_data)
    params = state.define_params(all_data['NPP'], priors_from, rel_err)
    residuals = state.define_residuals(params['Po']['prior'], gamma)
    state_elements = framework.define_state_elements(tracers, params, LAYERS)
    equation_elements = framework.define_equation_elements(tracers, LAYERS)
    xo = framework.define_prior_vector(tracers, residuals, params, LAYERS)
    Co = framework.define_cov_matrix(tracers, residuals, params, LAYERS)

    ati_results = ati.find_solution(
        tracers, state_elements, equation_elements, xo, Co, GRID, ZG,
        UMZ_START, MLD)
    xhat, Ckp1, *_ = ati_results
    x_resids = ati.normalized_state_residuals(xhat, xo, Co)
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, LAYERS)
    tracer_estimates, residual_estimates, param_estimates = estimates

    state.merge_by_keys(tracer_estimates, tracers)
    state.merge_by_keys(param_estimates, params)
    state.merge_by_keys(residual_estimates, residuals)

    residuals_sym = budgets.get_symbolic_residuals(
        residuals, UMZ_START, LAYERS)
    residual_estimates_by_zone = budgets.integrate_by_zone(
        residuals_sym, state_elements, Ckp1, residuals=residuals)
    state.merge_by_keys(residual_estimates_by_zone, residuals)

    inventories_sym = budgets.get_symbolic_inventories(
        tracers, UMZ_START, LAYERS, THICK)
    inventories = budgets.integrate_by_zone_and_layer(
        inventories_sym, state_elements, Ckp1, LAYERS, tracers=tracers)

    int_fluxes_sym = fluxes.get_symbolic_int_fluxes(
        UMZ_START, LAYERS, THICK, GRID, ZG, MLD)
    int_fluxes = budgets.integrate_by_zone_and_layer(
        int_fluxes_sym, state_elements, Ckp1, LAYERS, tracers=tracers,
        params=params)

    prior_estimates = unpack_state_estimates(
        tracers, params, state_elements, xo, Co, LAYERS)
    prior_tracers, _, prior_params = prior_estimates
    prior_fluxes = budgets.integrate_by_zone_and_layer(
        int_fluxes_sym, state_elements, Co, LAYERS, tracers=prior_tracers,
        params=prior_params)

    sink_fluxes = fluxes.sinking_fluxes(
        LAYERS, state_elements, Ckp1, tracers, params)

    production_profile = fluxes.production_profile(
        LAYERS, state_elements, Ckp1, tracers, params, GRID, mld=MLD)

    residence_times = timescales.calculate_residence_times(
        inventories_sym, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
        tracers, params, state_elements, Ckp1, ZONE_LAYERS, UMZ_START)

    turnover_times = timescales.calculate_turnover_times(
        inventories_sym, int_fluxes_sym, int_fluxes, tracers, params,
        state_elements, Ckp1, ZONE_LAYERS)

    calculate_B2(GRID, state_elements, Ckp1, tracers, params)

    to_pickle = {'tracers': tracers,
                 'params': params,
                 'residuals': residuals,
                 'inventories': inventories,
                 'int_fluxes': int_fluxes,
                 'sink_fluxes': sink_fluxes,
                 'production_profile': production_profile,
                 'residence_times': residence_times,
                 'turnover_times': turnover_times,
                 'state_elements': state_elements,
                 'equation_elements': equation_elements,
                 'x_resids': x_resids,
                 'prior_fluxes': prior_fluxes}

    save_path = f'../../results/exports/{priors_from}_{rel_err}_{gamma}.pkl'
    with open(save_path, 'wb') as file:
        dump(to_pickle, file)


if __name__ == '__main__':

    start_time = time()

    study_sites = ('NA', 'SP')
    gammas = (0.5, 1, 5, 10)
    rel_errs = (0.1, 0.2, 0.5, 1)

    with Pool() as p:
        p.starmap(run_model, product(study_sites, gammas, rel_errs))

    print(f'--- {(time() - start_time)/60} minutes ---')
