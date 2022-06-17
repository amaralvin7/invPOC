"""Run twin experiments to validate the inverse model."""
from pickle import dump
from time import time

import src.exports.data as data
import src.exports.state as state
import src.exports.twinexperiments as te
import src.framework as framework
import src.tools as tools
from src.ati import find_solution
from src.exports.constants import *
from src.unpacking import unpack_state_estimates


def run_twin_experiment(priors_from):
    """Run the twin experiment and pickle results.

    Args:
        priors_from (str): Location from which to pick B2p and Bm2 priors from.
        Can be NA (North Atlantic) or SP (Station P).
    """
    gamma = 0.5
    rel_err = 0.5
    targets = te.load_targets(priors_from, gamma, rel_err)

    all_data = data.load_data()
    state_elements = targets['state_elements']
    equation_elements = targets['equation_elements']

    tracers = te.generate_pseudodata(targets, priors_from, gamma, rel_err)
    params = state.define_params(all_data['NPP'], priors_from, rel_err)
    residuals = state.define_residuals(params['Po']['prior'], gamma)
    xo = framework.define_prior_vector(tracers, residuals, params, LAYERS)
    Co = framework.define_cov_matrix(tracers, residuals, params, LAYERS)

    ati_results = find_solution(
        tracers, state_elements, equation_elements, xo, Co, GRID, ZG,
        UMZ_START, mld=MLD)
    xhat, Ckp1, *_ = ati_results
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
        dump(to_pickle, file)


if __name__ == '__main__':

    start_time = time()

    run_twin_experiment('NA')
    run_twin_experiment('SP')

    print(f'--- {(time() - start_time)/60} minutes ---')
