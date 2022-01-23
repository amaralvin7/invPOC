#!/usr/bin/env python3
import time
import data.data as data
import data.parameters as parameters
import framework as fw
from unpacking import unpack_state_estimates
from ati import find_solution
import output

start_time = time.time()





all_data = data.load_data()
poc_data = data.process_poc_data(all_data['POC'])
tracers = data.define_tracers(poc_data)
params = parameters.define_params(all_data['NPP'], 'NA')
state_elements= fw.define_state_elements(tracers, params)
equation_elements = fw.define_equation_elements(tracers)
xo = fw.define_prior_vector(tracers, params)
Co = fw.define_cov_matrix(tracers, params)

xhat, Ckp1 = find_solution(tracers, state_elements, equation_elements, xo, Co)
unpack_state_estimates(tracers, params, state_elements, xhat, Ckp1)

output.write(params)







print(f'--- {(time.time() - start_time)/60} minutes ---')