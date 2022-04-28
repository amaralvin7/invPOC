import time
import pickle
from itertools import product
from multiprocessing import Pool
from numpy import diff
import sys
import src.exports.data as data
import src.exports.state as state
import src.framework as framework
import src.output as output
import src.budgets as budgets
import src.fluxes as fluxes
import src.timescales as timescales
from src.unpacking import unpack_state_estimates
from src.ati import find_solution

mld = 30  # mixed layer depth
zg = 100  # grazing zone depth
grid = (30, 50, 100, 150, 200, 330, 500)
umz_start = grid.index(zg) + 1
layers = tuple(range(len(grid)))
zone_layers = ('EZ', 'UMZ') + layers
thick = diff((0,) + grid)

def run_model(priors_from, gamma, rel_err):

    all_data = data.load_data()
    poc_data = data.process_poc_data(all_data['POC'], grid)
    tracers = state.define_tracers(poc_data)
    params = state.define_params(all_data['NPP'], priors_from, rel_err)
    residuals = state.define_residuals(params['Po']['prior'], gamma, mld)
    state_elements = framework.define_state_elements(tracers, params, layers)
    equation_elements = framework.define_equation_elements(tracers, layers)
    xo = framework.define_prior_vector(tracers, residuals, params, layers)
    Co = framework.define_cov_matrix(tracers, residuals, params, layers)

    ati_results = find_solution(
        tracers, state_elements, equation_elements, xo, Co, grid, zg, mld,
        True, umz_start, priors_from, None)  # last 2 args just for debugging GT inversions, delete later
    xhat, Ckp1, _, _ = ati_results
    x_resids = output.normalized_state_residuals(xhat, xo, Co)
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, layers)
    tracer_estimates, residual_estimates, param_estimates = estimates

    output.merge_by_keys(tracer_estimates, tracers)
    output.merge_by_keys(param_estimates, params)
    output.merge_by_keys(residual_estimates, residuals)

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
    
    production_profile = fluxes.production_prof(
        layers, state_elements, Ckp1, tracers, params, mld, grid)

    residence_times = timescales.calculate_residence_times(
        inventories_sym, int_fluxes_sym, int_fluxes, residuals_sym, residuals,
        tracers, params, state_elements, Ckp1, zone_layers, umz_start)

    turnover_times = timescales.calculate_turnover_times(
        inventories_sym, int_fluxes_sym, int_fluxes, tracers, params,
        state_elements, Ckp1, zone_layers)
    
    output.calculate_B2(grid, state_elements, Ckp1, tracers, params)
    
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
                 'x_resids': x_resids
                 }
    
    save_path = f'../../results/exports/{priors_from}_{rel_err}_{gamma}.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)

if __name__ == '__main__':

    start_time = time.time()

    study_sites = ('NA', 'SP')
    gammas = (0.5, 1, 5, 10)
    rel_errs = (0.1, 0.2, 0.5, 1)

    pool = Pool()
    pool.starmap(run_model, product(study_sites, gammas, rel_errs))

    print(f'--- {(time.time() - start_time)/60} minutes ---')