#!/usr/bin/env python3
import numpy as np

import statsmodels.formula.api as smf

from src.constants import MMC, DPY, MLD

def define_tracers(data):
    
    tracers = {'POCS': {}, 'POCL': {}}
    
    for t in tracers:
        tracers[t]['prior'] = data[t]
        tracers[t]['prior_e'] = data[f'{t}_se']
    
    return tracers

def define_residuals(proportional_to, gamma):
    
    residuals = {'POCS': {}, 'POCL': {}}
    
    for tracer in residuals:
        residuals[tracer]['prior'] = 0
        residuals[tracer]['prior_e'] = gamma * proportional_to * MLD
    
    return residuals

def define_params(npp_data, priors_from, rel_err):
    
    params = {}
    
    B2p_prior, B2p_error, Bm2_prior, Bm2_error = contextual_priors(priors_from)
    Po_prior, Po_error, Lp_prior, Lp_error = npp_priors(npp_data)

    params['ws'] = set_prior(2, 2*rel_err)
    params['wl'] = set_prior(20, 20*rel_err)
    params['B2p'] = set_prior(B2p_prior*MMC/DPY, B2p_error*MMC/DPY)
    params['Bm2'] = set_prior(Bm2_prior/DPY, Bm2_error/DPY)
    params['Bm1s'] = set_prior(0.1, 0.1*rel_err)
    params['Bm1l'] = set_prior(0.15, 0.15*rel_err)
    params['Po'] = set_prior(Po_prior, Po_error, depth_varying=False)
    params['Lp']= set_prior(Lp_prior, Lp_error, depth_varying=False)
    params['B3'] = set_prior(0.06, 0.06*rel_err, depth_varying=False)
    params['a'] = set_prior(0.3, 0.15, depth_varying=False)
    params['zm'] = set_prior(500, 250, depth_varying=False)        
    
    return params

def set_prior(prior, error, depth_varying=True):
    
    data = {}
    
    data['prior'] = prior
    data['prior_e'] = error
    data['dv'] = depth_varying
    
    return data

def contextual_priors(priors_from):

    if priors_from == 'NA':  # Murnane et al. 1996, DSR
        B2p_prior = (2/21) # m^3 mg^-1 y^-1
        B2p_error = np.sqrt((0.2/21)**2 + (-1*(2/21**2))**2)
        Bm2_prior = 156  # y^-1
        Bm2_error = 17
    else:  # Murnane 1994, JGR
        B2p_prior = (0.8/1.57) # m^3 mg^-1 y^-1
        B2p_error = np.sqrt((0.9/1.57)**2 + (-0.48*(0.8/1.57**2))**2)
        Bm2_prior = 400  # y^-1
        Bm2_error = 10000
    
    return B2p_prior, B2p_error, Bm2_prior, Bm2_error

def npp_priors(npp_data):
    
        npp_data_clean = npp_data.loc[(npp_data['NPP'] > 0)]

        MIXED_LAYER_UPPER_BOUND, MIXED_LAYER_LOWER_BOUND = 28, 35

        npp_mixed_layer = npp_data_clean.loc[
            (npp_data_clean['target_depth'] >= MIXED_LAYER_UPPER_BOUND) &
            (npp_data_clean['target_depth'] <= MIXED_LAYER_LOWER_BOUND)]

        npp_below_mixed_layer = npp_data_clean.loc[
            npp_data_clean['target_depth'] >= MIXED_LAYER_UPPER_BOUND]

        Po_prior = npp_mixed_layer['NPP'].mean()/MMC
        Po_prior_e = npp_mixed_layer['NPP'].sem()/MMC

        npp_regression = smf.ols(
            formula='np.log(NPP/(Po_prior*MMC)) ~ target_depth',
            data=npp_below_mixed_layer).fit()

        Lp_prior = -1/npp_regression.params[1]
        Lp_prior_e = npp_regression.bse[1]/npp_regression.params[1]**2

        return Po_prior, Po_prior_e, Lp_prior, Lp_prior_e