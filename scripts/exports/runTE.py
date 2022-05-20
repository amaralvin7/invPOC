import pickle
import time
import pandas as pd
import sys
import src.exports.twinexperiments as te
import src.exports.state as state
import src.framework as framework
import src.tools as tools
from src.unpacking import unpack_state_estimates
from src.ati import find_solution
from src.exports.constants import *

def run_twin_experiment(priors_from, gamma, rel_err):

    targets = te.load_targets(priors_from, gamma, rel_err)

    all_data = pd.read_excel('../../data/exports.xlsx', sheet_name=None)
    state_elements = targets['state_elements']
    equation_elements = targets['equation_elements']

    tracers = te.generate_pseudodata(targets, priors_from, gamma, rel_err)
    params = state.define_params(all_data['NPP'], priors_from, rel_err)
    residuals = state.define_residuals(params['Po']['prior'], gamma)
    xo = framework.define_prior_vector(tracers, residuals, params, LAYERS)
    Co = framework.define_cov_matrix(tracers, residuals, params, LAYERS)

    ati_results = find_solution(
        tracers, state_elements, equation_elements, xo, Co, GRID, ZG,
        UMZ_START, priors_from, None, mld=MLD)  # last 2 args before MLD are just for debugging GT inversions, delete later
    xhat, Ckp1, _, _ = ati_results
    estimates = unpack_state_estimates(
        tracers, params, state_elements, xhat, Ckp1, LAYERS)
    tracer_estimates, residual_estimates, param_estimates = estimates
    tools.merge_by_keys(tracer_estimates, tracers)
    tools.merge_by_keys(param_estimates, params)
    tools.merge_by_keys(residual_estimates, residuals)

    to_pickle = {'tracers': tracers, 'params': params, 'residuals': residuals,
                 'targets': targets}
    
    save_path = f'../../results/exports/{priors_from}_{rel_err}_{gamma}_TE.pkl'
    with open(save_path, 'wb') as file:
                pickle.dump(to_pickle, file)


if __name__ == '__main__':

    start_time = time.time()

    run_twin_experiment('NA', 0.5, 0.5)
    run_twin_experiment('SP', 0.5, 0.5)

    print(f'--- {(time.time() - start_time)/60} minutes ---')