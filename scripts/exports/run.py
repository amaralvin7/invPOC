#!/usr/bin/env python3
import time

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


start_time = time.time()

all_data = data.load_data()
poc_data = data.process_poc_data(all_data['POC'])
tracers = state.define_tracers(poc_data)
params = state.define_params(all_data['NPP'], 'NA')
residuals = state.define_residuals(params['Po']['prior'])
state_elements = framework.define_state_elements(tracers, params, LAYERS)
equation_elements = framework.define_equation_elements(tracers, LAYERS)
xo = framework.define_prior_vector(tracers, residuals, params, LAYERS)
Co = framework.define_cov_matrix(tracers, residuals, params, LAYERS)

ati_results = find_solution(
    tracers, state_elements, equation_elements, xo, Co, GRID, ZG, True)
xhat, Ckp1, convergence_evolution, cost_evolution = ati_results
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

output.write_output(
    params, residuals, inventories, int_fluxes,residence_times, turnover_times)

print(f'--- {(time.time() - start_time)/60} minutes ---')