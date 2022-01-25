#!/usr/bin/env python3
import time
import data.data as data
import data.parameters as parameters
import framework as fw
from unpacking import unpack_state_estimates
from ati import find_solution
import output
import budgets

start_time = time.time()





all_data = data.load_data()
poc_data = data.process_poc_data(all_data['POC'])
tracers = data.define_tracers(poc_data)
params = parameters.define_params(all_data['NPP'], 'NA')
residuals = data.define_residuals(params)
state_elements= fw.define_state_elements(tracers, params)
equation_elements = fw.define_equation_elements(tracers)
xo = fw.define_prior_vector(tracers, residuals, params)
Co = fw.define_cov_matrix(tracers, residuals, params)

ati_results = find_solution(tracers, state_elements, equation_elements, xo, Co)
xhat, Ckp1, convergence_evolution, cost_evolution = ati_results
unpack_state_estimates(tracers, residuals, params, state_elements, xhat, Ckp1)

residuals_sym = budgets.get_symbolic_residuals(residuals)
budgets.integrate_residuals_by_zone(
    residuals, residuals_sym, state_elements, tracers, params, Ckp1)

output.write(params, residuals)







print(f'--- {(time.time() - start_time)/60} minutes ---')